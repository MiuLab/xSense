import torch
import torch.nn as nn
from torch import optim
import numpy as np
import os
from tqdm import tqdm
from load import loadTrainData
from utils import *
from constants import BOS_IDX

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
torch.manual_seed(200)


def train(args):

    dataloader = loadTrainData(args)

    decoder_optimizer, decoder, mask_generator, spine = build_model_optimizer(args)
    losses = np.zeros(4)
    for epoch in tqdm(range(1, args.epoch+1)):
        for i, (trg_emb, ctx_emb, def_ids, length) in enumerate(dataloader):
            trg_emb = trg_emb.to(device)
            ctx_emb = ctx_emb.to(device)
            def_ids = def_ids.to(device)
            length = length.to(device)

            mask, max_target_len = get_mask(length)
            mask = mask.to(device)
            # (seq, batch)
            mask = mask.transpose(0, 1) 
            def_ids = def_ids.transpose(0, 1)

            sp_z, sp_w, loss_terms = spine(trg_emb)
            sense_vec, _, _ = mask_generator(sp_z, sp_w, ctx_emb)
            sense_vec = sense_vec.unsqueeze(0)

            decoder_input = torch.LongTensor([[BOS_IDX] * args.batch_size]).to(device)
            decoder_hidden = trg_emb.unsqueeze(0)     
            decoder_hidden2 = sense_vec

            decoder_optimizer.zero_grad()
            loss = 0
            n_totals = 0
            s2s_sum = 0
            # teacher-forcing
            for t in range(max_target_len):
                decoder_output, decoder_hidden, decoder_hidden2 = decoder(decoder_input, decoder_hidden, decoder_hidden2, sense_vec)
                decoder_input = def_ids[t].unsqueeze(0) # Next input is current target
                
                mask_loss, nTotal = maskNLLLoss(decoder_output, def_ids[t], mask[t])
                s2s_sum += mask_loss.item() * nTotal
                n_totals += nTotal

                loss += mask_loss
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 50.0)

            decoder_optimizer.step()

            # print log
            reconstruction_loss, psl_loss, asl_loss = loss_terms
            losses[0] += reconstruction_loss.item()
            losses[1] += asl_loss.item()
            losses[2] += psl_loss.item()
            losses[3] += s2s_sum / n_totals
                    
            if (i+1) % args.print_every == 0:
                print_loss = 0
                losses /= args.print_every
                print("Rec Loss = %.4f, ASL = %.4f, PSL = %.4f, S2S = %.4f, Sparsity = %.2f"
                                        %(*losses, compute_sparsity(sp_z.cpu().data.numpy())))
                losses = np.zeros(4)

        # save current model
        if epoch == args.epoch or epoch % save_every == 0:
            torch.save({
                'decoder': decoder.state_dict(),
                'spine': spine.state_dict(),
                'mask_gen': mask_generator.state_dict(),
                'decoder_opt': decoder_optimizer.state_dict(),
                'loss': loss
            }, os.path.join(args.save_dir, 'model', 'xSense_{}.tar'.format(epoch)))

        