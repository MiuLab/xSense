import torch
import random
import pickle
import os
import numpy as np
from train import indexesFromSentence, wordVar
from load import SOS_token, EOS_token
from load import DEC_MAX_LENGTH, loadPrepareData, Voc, UNK_token
from model_wic import *
import nltk
import visualize_online
from config import save_dir
import math

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

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
    sense_vec, topId  = mask_generator(h, sp_w, mapped_ctx)

    return sense_vec, topId


def evaluateTestFile(decoder, spine, mask_generator, sif_table, mapping, voc, voc_dec, batch_size, reverse, testfile, outputfile, vectors, top_k_words):
    with open(testfile,'r') as f:
        content = f.readlines()
    out_file = open(outputfile,'w')
    unkId = []
    pairs = []
    for idx, word in enumerate(content):
        word = word.strip()
        if word not in voc.word2index:
            unkId.append(idx)
            continue
        pairs.append([word, idx])

    l = (len(pairs) // batch_size)+1
    SV = []
    ID = []
    selecID, ans = [], []
    for i in range(l):
        batch_pair = pairs[i*batch_size:(i+1)*batch_size]
        sv, topId = evaluateBatch(decoder, spine, mask_generator, sif_table, mapping, voc, voc_dec, len(batch_pair), batch_pair)

        for v in sv.data.cpu().numpy():
            SV.append(v)
        for idx in topId.data.cpu().numpy():
            ID.append(idx)
    
    i = 0
    for idx in range(len(content)):
        if idx in unkId:
            selecID.append(0) # Guess
        else:
            selecID.append(ID[i])
            i += 1
            
    # re-arrange answer to output format
    for idx in range(0, len(selecID), 2):
        if selecID[idx+1] == selecID[idx]:
            ans.append('T')
        else:
            ans.append('F')

    gold = open('data/wic_ans.txt').read().splitlines()
    total = len(gold)
    correct = 0    
    for g, a in zip(gold, ans):
        if g.strip() == a:
            correct += 1
    
    assert len(ans) == total
    
    acc = round(correct/total, 4)
    print("---------------------------------------------")
    print ('acc:', acc)
    print ('T:', ans.count('T'))
    print ('F:', ans.count('F'))
    print("---------------------------------------------")

    
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
    decoder.eval()

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

