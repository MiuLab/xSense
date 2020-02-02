import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
from load import loadTrainData, loadPretrainData
from utils import *
from constants import BOS_IDX, PAD_IDX

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
torch.manual_seed(200)


def train(args):

    dataloader = loadTrainData(args)

    decoder_optimizer, decoder, mask_generator, spine = build_model_optimizer(args)
    log_loss = np.zeros(4)
    for epoch in tqdm(range(1, args.epoch+1)):
        for i, (trg_emb, ctx_emb, def_ids, length) in enumerate(dataloader):
            trg_emb = trg_emb.to(device)
            ctx_emb = ctx_emb.to(device)
            def_ids = def_ids.to(device)
            length = length.to(device)

            # (seq, batch)
            def_ids = def_ids.transpose(0, 1)

            sp_z, sp_w, loss_terms = spine(trg_emb)
            aligned_ctx, sense_vec, _, _ = mask_generator(sp_z, sp_w, ctx_emb)
        
            decoder_input = torch.LongTensor([[BOS_IDX] * args.batch_size]).to(device)
            decoder_hidden = trg_emb.unsqueeze(0)     
            decoder_hidden2 = aligned_ctx.unsqueeze(0)

            decoder_optimizer.zero_grad()
            # teacher-forcing
            losses = torch.zeros(args.batch_size).to(device)
            max_len = torch.max(length).item() # max length in the current batch
            for t in range(max_len):
                decoder_output, decoder_hidden, decoder_hidden2 = \
                    decoder(decoder_input, decoder_hidden, decoder_hidden2, sense_vec.unsqueeze(0))
                decoder_input = def_ids[t].unsqueeze(0) # Next input is current target
                losses += F.cross_entropy(decoder_output, def_ids[t], ignore_index=PAD_IDX, reduction='none')
 
            loss = torch.mean(losses / length.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 50.0)

            decoder_optimizer.step()

            # print log
            reconstruction_loss, psl_loss, asl_loss = loss_terms
            log_loss[0] += reconstruction_loss.item()
            log_loss[1] += asl_loss.item()
            log_loss[2] += psl_loss.item()
            log_loss[3] += loss.item()
                    
            if (i+1) % args.print_every == 0:
                print_loss = 0
                log_loss /= args.print_every
                print("Rec Loss = %.4f, ASL = %.4f, PSL = %.4f, S2S = %.4f, Sparsity = %.2f"
                                        %(*log_loss, compute_sparsity(sp_z.cpu().data.numpy())))
                log_loss = np.zeros(4)

        # save current model
        if epoch == args.epoch or epoch % args.save_every == 0:
            torch.save({
                'decoder': decoder.state_dict(),
                'spine': spine.state_dict(),
                'mask_gen': mask_generator.state_dict(),
                'decoder_opt': decoder_optimizer.state_dict(),
                'loss': loss
            }, os.path.join(args.save_dir, 'model', 'xSense_{}.tar'.format(epoch)))


def pretrain_spine(args):
    dataloader = loadPretrainData(args)
    optimizer, spine = build_model_optimizer(args)
    best_score = 0
    for epoch in tqdm(range(1, args.epoch+1)):
        for i, trg_emb in enumerate(dataloader):
            trg_emb = trg_emb[0].to(device)
            optimizer.zero_grad()
            sp_z, sp_w, loss_terms = spine(trg_emb)
            rec_loss, psl_loss, asl_loss = loss_terms
            loss = 7*rec_loss + psl_loss + asl_loss
            loss.backward()
            optimizer.step()
            spine.orthogonalize()

        sparsity = compute_sparsity(sp_z.cpu().data.numpy())
        print("Rec Loss = %.4f, ASL = %.4f, PSL = %.4f, Sparsity = %.2f"
            %(rec_loss, asl_loss, psl_loss, sparsity))
        
        if sparsity > best_score:
            best_score = sparsity
            torch.save({
                'epoch': epoch,
                'sparsity': sparsity,
                'spine': spine.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(args.save_dir, 'model', 'pretrained_spine.tar'))
        
