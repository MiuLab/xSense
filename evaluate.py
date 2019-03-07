import torch
import random
import pickle
import os
import numpy as np
from train import indexesFromSentence, wordVar
from load import SOS_token, EOS_token
from load import DEC_MAX_LENGTH, loadPrepareData, Voc, UNK_token
from model import *
import visualize_online
from config import save_dir
import math

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


def decode(decoder, mapped_ctx, masked_h, trg_emb, voc, batch_size, input_words, max_length=DEC_MAX_LENGTH):

    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]]).to(device)
    decoder_hidden = trg_emb.unsqueeze(0)
    decoder_hidden2 = mapped_ctx.unsqueeze(0)
    
    decoded_words = []
    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_hidden2 = decoder(
            decoder_input, decoder_hidden, decoder_hidden2, masked_h
        )
        _, topi = decoder_output.topk(3)
        ni = torch.cat((ni, topi[:,0].unsqueeze(1)), dim=1) if di else topi[:,0].unsqueeze(1)
        
        decoder_input = topi[:,0].unsqueeze(0)
        decoder_input = decoder_input.to(device)

    for i in range(ni.shape[0]):
        words = []
        for j in range(ni.shape[1]):
            n = ni[i][j].item()
            if n == EOS_token:
                words.append('<EOS>')
                break
            else:
                words.append(voc.index2word[n])
        decoded_words.append(words)

    return decoded_words


def evaluateBatch(decoder, spine, mask_generator, sif_table, mapping, voc, voc_dec, batch_size, batch_pair):
    input_batch, input_words = [], []
    for pair in batch_pair:
        input_words.append(pair[0])
        input_batch.append(pair[1])
    input_batch = torch.LongTensor(input_batch)
    input_words = wordVar(input_words, voc)
    out, h, sp_w, trg_emb, loss, loss_terms = spine(input_words)
    encoded_ctx = sif_table(input_batch)
    mapped_ctx = mapping(encoded_ctx)
    masked_h, attn, indices = mask_generator(h, sp_w, mapped_ctx)

    return decode(decoder, mapped_ctx, masked_h, trg_emb, voc_dec, batch_size, input_words), attn, indices

def evaluateTestFile(decoder, spine, mask_generator, sif_table, mapping, voc, voc_dec, batch_size, reverse, testfile, outputfile, vectors, top_k_words):
    with open(testfile,'r') as f:
        content = f.readlines()
    out_file = open(outputfile,'w')
    pairs = []
    for idx, x in enumerate(content):
        x = x.strip()
        pair = x.split(';')
        if pair[0].strip() not in voc.word2index:
            continue
        pairs.append([pair[0].strip(), idx, pair[2].strip(), pair[1].strip()])
    l = (len(pairs) // batch_size)+1
    
    for i in range(l):
        batch_pair = pairs[i*batch_size:(i+1)*batch_size]
        output_words, attn, indices = evaluateBatch(decoder, spine, mask_generator, sif_table, mapping, voc, voc_dec, len(batch_pair), batch_pair)
        
        for bidx, (p, ass, idxs, ow) in enumerate(zip(batch_pair, attn, indices, output_words)):
            word = p[0]
            ctx = p[3]
            assert word in vectors, "word not found"

            print ('target word is: {}'.format(word))
            ctx = ctx.split()
            tidx = ctx.index(word)
            ctx = ctx[:tidx] + [ctx[tidx].upper()] + ctx[tidx+1:]
            ctx = ' '.join(ctx)
            print ('context is: {}'.format(ctx))
            print ('ground truth is: {}'.format(batch_pair[bidx][2]))
            print ('generated definition is: {}'.format(' '.join(ow)))
            ass = ass.squeeze(1).cpu().data.numpy()
            idxs = idxs.cpu().data.numpy()
            saidxs = np.argsort(ass)[::-1]

            for saidx in saidxs:
                a = ass[saidx]
                idx = idxs[saidx]
                print ('attn value for dim {} is: {}'.format(idx, a))
                print (top_k_words[idx])
            print ()
            
        for index in range(len(output_words)):
            w = batch_pair[index][0]
            ctx = batch_pair[index][1]
            truth = batch_pair[index][2].split(' ')
            pre = output_words[index]
            out_file.write('{} ; {} ; {} ; {}\n'.format(w, ctx, ' '.join(truth), ' '.join(pre)))
            
    out_file.close()

def runTest(n_layers, hidden_size, reverse, modelFile, corpus, batch_size, testfile, outputfile, sif):
    torch.set_grad_enabled(False)

    voc = torch.load('save/training_data/my_s2s_sent/voc.tar')
    voc_dec = torch.load('save/training_data/my_s2s_sent/voc_s2s.tar')
    pairs = torch.load('save/training_data/my_s2s_sent/pairs.tar')
    pretrained = torch.load('save/training_data/my_s2s_sent/pretrained.tar')
    sif_data = []
    for line in open(sif):
        line = line.strip().split()
        sif_data.append([float(i) for i in line])

    embedding = nn.Embedding.from_pretrained(torch.FloatTensor(pretrained))
    sif_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(sif_data))
    dec_emb = nn.Embedding.from_pretrained(torch.randn(voc_dec.n_words, 300), freeze=False)
    spine = SPINEModel(embedding)
    sif_table = SIFTable(sif_embedding)
    mask_generator = MaskGenerator(1000, 300)
    mapping = Mapping(enc_dim=300)
    decoder = DecoderRNN(dec_emb, hidden_size, voc_dec.n_words, n_layers)
    
    checkpoint = torch.load(modelFile)
    decoder.load_state_dict(checkpoint['de'])
    spine.load_state_dict(checkpoint['sp'])
    mask_generator.load_state_dict(checkpoint['msk'])
    mapping.load_state_dict(checkpoint['map'])
    decoder.eval() # effect only on dropout, batchNorm

    sif_table = sif_table.to(device)
    dec_emb = dec_emb.to(device)
    decoder = decoder.to(device)
    spine = spine.to(device)
    mask_generator = mask_generator.to(device)
    mapping = mapping.to(device)
    
    def dump_vectors(X, outfile, words, voc):
        print ("shape", X.shape)
        assert len(X) == len(words) #TODO print error statement
        if len(X) != len(words):
            print ('some words are oov, we keep only those existed in oxford dictionary')
        fw = open(outfile, 'w')
        for i in range(len(X)):
            fw.write(voc.index2word[words[i]] + " ")
            for j in X[i]:
                fw.write(str(j) + " ")
            fw.write("\n")
        fw.close()
    
    # dump spine embeddings
    if not os.path.exists('word2vec_emb_out'):
        dump_batch = 512
        ret = []
        print_words = list(range(voc.n_words))
        for i in range(math.ceil(voc.n_words/dump_batch)):
            ws = print_words[i*dump_batch:(i+1)*dump_batch]
            _, sp_z, _, _, _, _ = spine(ws)
            ret.extend(sp_z.cpu().data.numpy())
        dump_vectors(np.array(ret), 'word2vec_emb_out', print_words, voc)
    if not os.path.exists('vectors.pkl') or not os.path.exists('top_k_words.pkl'):
        visualize_online.load_vectors('word2vec_emb_out')
    with open('vectors.pkl','rb') as f:
        vectors = pickle.load(f)
    with open('top_k_words.pkl','rb') as f:
        top_k_words = pickle.load(f)
    #visualize_online.load_top_dimensions(10)
    
    evaluateTestFile(decoder, spine, mask_generator, sif_table, mapping, voc, voc_dec, batch_size, reverse, testfile, outputfile, vectors, top_k_words)

def find_top_closest_words(word ,spine ,voc ,k):
    if not os.path.exists('vectors.pkl'):
        visualize_online.load_vectors('word2vec_emb_out')
    with open('vectors.pkl','rb') as f:
        vectors = pickle.load(f)
    with open('top_k_words.pkl','rb') as f:
        top_k_words = pickle.load(f)
   
    input_word = [word]
    input_word = wordVar(input_word, voc)

    out, h, sp_w, loss, loss_terms = spine(input_word)
    h = h.cpu().numpy()[0].tolist()
    temp = [(j, i) for i, j in enumerate(h)]
    answer = []
    print (" -----------------------------------------------------")
    print ("Word of interest = " , word)
    for i, j in sorted(temp, reverse=True)[:k]:
        
        print('%s Top 10 closet word in dimension %d :'%(word, j))
        print([k[1] for k in top_k_words[j]])
        answer.append([k[1] for k in top_k_words[j]])
    return

