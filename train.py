import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torch.backends.cudnn as cudnn
import numpy as np
import itertools
import random
import math
import os
from tqdm import tqdm
from load import loadPrepareData
from load import SOS_token, EOS_token, PAD_token, UNK_token
from model import DecoderRNN, SPINEModel, MaskGenerator, Mapping, SIFTable
from config import save_dir
import numpy as np

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

cudnn.benchmark = True

#############################################
# generate file name for saving parameters
#############################################
def filename(reverse, obj):
	filename = ''
	if reverse:
		filename += 'reverse_'
	filename += obj
	return filename


#############################################
# Prepare Training Data
#############################################
def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] if word in voc.word2index else UNK_token for word in sentence.split(' ')]  + [EOS_token]

# batch_first: true -> false, i.e. shape: seq_len * batch
def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

# convert to index, add EOS, zero padding
# return output variable, mask, max length of the sentences in batch
def outputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.ByteTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len

def wordVar(input_words, voc):
    return [voc.word2index[w] for w in input_words]

# pair_batch is a list of (input, output) with length batch_size
# sort list of (input, output) pairs by input length, reverse input
# return input, lengths for pack_padded_sequence, output_variable, mask
def batch2TrainData(voc, voc_dec, pairs, batch_size, reverse):
    #[random.choice(pairs) for _ in range(batch_size)]
    pair_batch = []
    while len(pair_batch) < batch_size:
        temp = random.choice(pairs)
        if temp[0] in voc.word2index:
            pair_batch.append(temp)

    if reverse:
        pair_batch = [pair[1:][::-1] for pair in pair_batch] # pair[0] is target word

    input_batch, output_batch, input_words = [], [], []

    for pair in pair_batch:
        if pair[0] in voc.word2index:
            length = len(pair[0])
            input_words.append(pair[0])
            input_batch.append(pair[1])
            output_batch.append(pair[2])
            
    output, mask, max_target_len = outputVar(output_batch, voc_dec)
    input_words = wordVar(input_words, voc)
    input_batch = torch.LongTensor(input_batch)
    
    return output, mask, max_target_len, input_words, input_batch

def compute_sparsity(X):
    non_zeros = 1. * np.count_nonzero(X)
    total = X.size
    sparsity = 100. * (1 - (non_zeros)/total)
    return sparsity

#############################################
# Training
#############################################

def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    s2s_loss = torch.nn.NLLLoss()
    crossEntropy = s2s_loss(inp, target)
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()

def clip_parameters(model, clip):
    """
    Clip model weights.
    """
    if clip > 0:
        for x in model.parameters():
            x.data.clamp_(-clip, clip)


def train(iteration, target_variable, mask, max_target_len, input_words, spine, contex, \
            decoder, mask_generator, sif_table, mapping, decoder_optimizer, batch_size):
    
    decoder_optimizer.zero_grad()
    
    target_variable = target_variable.to(device)
    mask = mask.to(device)
   
    loss = 0
    print_losses = []
    n_totals = 0
    batch_loss = np.zeros(4)
    
    out, sp_z, sp_w, trg_emb, loss, loss_terms = spine(input_words)

    encoded_ctx = sif_table(contex) # [bs, 300]     
    mapped_ctx = mapping(encoded_ctx)
    masked_h, _, _ = mask_generator(sp_z, sp_w, mapped_ctx)

    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)
    decoder_hidden = trg_emb.unsqueeze(0)     # candidates: encoded_ctx #masked_h #mapped_ctx
    decoder_hidden2 = mapped_ctx.unsqueeze(0) # candidates: encoded_ctx
    
    # teacher-forcing
    for t in range(max_target_len):
        decoder_output, decoder_hidden, decoder_hidden2 = decoder(decoder_input, decoder_hidden, decoder_hidden2, masked_h)
        
        decoder_input = target_variable[t].view(1, -1) # Next input is current target
        
        mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
        loss += mask_loss
        print_losses.append(mask_loss.item() * nTotal) #v0.4
        n_totals += nTotal
        
    loss.backward()

    clip = 50.0
    _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    decoder_optimizer.step()

    s2s_loss = sum(print_losses) / n_totals
    reconstruction_loss, psl_loss, asl_loss = loss_terms
    batch_loss[0]+=reconstruction_loss.item()
    batch_loss[1]+=asl_loss.item()
    batch_loss[2]+=psl_loss.item()
    batch_loss[3]+=s2s_loss

    return batch_loss, sp_z # sp_z is for computing sparsity


def trainIters(corpus, wordvec, word_f, sif, reverse, n_iteration, learning_rate, batch_size, n_layers, hidden_size,
                print_every, save_every, dropout, decoder_learning_ratio=5.0):

    voc, voc_dec, pairs, pretrained, sif_pretrained = loadPrepareData(corpus, wordvec, word_f, sif)

    # Get training data
    corpus_name = os.path.split(corpus)[-1].split('.')[0]
    training_batches = None
    
    try:
        training_batches = torch.load(os.path.join(save_dir, 'training_data', corpus_name,
                                                   '{}_{}_{}.tar'.format(n_iteration, \
                                                                         filename(reverse, 'training_batches'), \
                                                                         batch_size)))
    except FileNotFoundError:
        print('Training pairs not found, generating ...')

        training_batches = [batch2TrainData(voc, voc_dec, pairs, batch_size, reverse) for _ in range(n_iteration)]
        
        torch.save(training_batches, os.path.join(save_dir, 'training_data', corpus_name, '{}_{}_{}.tar' \
                                    .format(n_iteration, filename(reverse, 'training_batches'), batch_size)))

    # model
    checkpoint = None
    print('Building decoder ...')
    embedding = nn.Embedding.from_pretrained(torch.FloatTensor(pretrained))
    spine = SPINEModel(embedding)
    sp_ckpt = torch.load('save/model/pretrained_spine.tar')                                                                      
    spine.load_state_dict(sp_ckpt['sp'])
    
    sif_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(sif_pretrained))
    sif_table = SIFTable(sif_embedding)
    dec_emb = nn.Embedding.from_pretrained(torch.randn(voc_dec.n_words, 300), freeze=False)
    decoder = DecoderRNN(dec_emb, hidden_size, voc_dec.n_words, n_layers, dropout)
    mask_generator = MaskGenerator(z_dim=1000, enc_dim=300)
    mapping = Mapping(enc_dim=300) 

    # use cuda
    sif_table = sif_table.to(device)
    dec_emb = dec_emb.to(device)
    decoder = decoder.to(device)
    spine = spine.to(device)
    mask_generator = mask_generator.to(device)
    mapping = mapping.to(device)

    # optimizer
    print('Building optimizers ...')
    decoder_optimizer = optim.Adam(filter(lambda p: p.requires_grad, list(decoder.parameters())+list(mask_generator.parameters())+\
                                    list(mapping.parameters())), lr=learning_rate * decoder_learning_ratio)

    # initialize
    print('Initializing ...')
    start_iteration = 1
    perplexity = []
    print_loss = 0 
    batch_losses = np.zeros(4)
    print_words = set()
    for iteration in tqdm(range(start_iteration, n_iteration + 1)):
        training_batch = training_batches[iteration - 1]

        target_variable, mask, max_target_len, input_words, contex = training_batch

        print_words.update(input_words)
        
        batch_loss, sp_z = train(iteration, target_variable, mask, max_target_len, input_words, spine, contex, \
                                    decoder, mask_generator, sif_table, mapping, decoder_optimizer, batch_size)
        
        loss = batch_loss[-1]
        print_loss += loss
        perplexity.append(loss)
        batch_losses += batch_loss
	
        if iteration % print_every == 0:
            print_loss_avg = math.exp(print_loss / print_every)
            print_loss = 0
            batch_losses /= print_every
            print("Reconstruction Loss = %.4f, ASL = %.4f,"\
                        " PSL = %.4f, S2S = %.4f, Sparsity = %.2f"
                        %(*batch_losses, compute_sparsity(sp_z.cpu().data.numpy())))
            batch_losses = np.zeros(4)

        if (iteration % save_every == 0 and iteration > 45000):
            directory = os.path.join(save_dir, 'model', corpus_name, '{}-{}_{}'.format(n_layers, n_layers, hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'de': decoder.state_dict(),
                'sp': spine.state_dict(),
                'msk': mask_generator.state_dict(),
                'map': mapping.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'plt': perplexity
            }, os.path.join(directory, '{}_{}.tar'.format(iteration, filename(reverse, 'backup_bidir_model'))))


