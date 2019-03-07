import torch
import re
import os
import unicodedata
from collections import Counter
from config import ENC_MAX_LENGTH, DEC_MAX_LENGTH, save_dir
import itertools

SOS_token = 0
EOS_token = 1
PAD_token = 2
UNK_token = 3

class Voc:
    def __init__(self, name, words):
        self.name = name
        self.word2index = {}
        if name == 's2s':
            self.index2word = {0: "SOS", 1: "EOS", 2:"PAD", 3:"UNK"}
        else:
            self.index2word = {}
        self.n_words = len(self.index2word)
        for k, v in self.index2word.items():
            self.word2index[v] = k
        for w in words:
            self.word2index[w] = len(self.word2index)
            self.index2word[len(self.index2word)] = w
            self.n_words += 1

def readVocs(corpus, corpus_name, words, word_f):
    print("Reading lines...")

    # combine every two lines into pairs and normalize
    with open(corpus, encoding='utf-8') as f:
        content = f.readlines()
    # open corresponding words for spine
    with open(word_f, encoding='utf-8') as wordfile:
        target_words = wordfile.read().splitlines()

    lines = [x.strip() for x in content]
    it = iter(lines)
    pairs = [[t, idx, next(it)] for idx, (t, x) in enumerate(zip(target_words, it))]

    voc = Voc(corpus_name, words)
    return voc, pairs

def filterPair(p):
    # input sequences need to preserve the last word for EOS_token
    #return len(p[1].split(' ')) < ENC_MAX_LENGTH and len(p[2].split(' ')) < DEC_MAX_LENGTH
    return len(p[2].split(' ')) < DEC_MAX_LENGTH

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(corpus, corpus_name, wordvec, word_f, sif):
    # read pretrained word2vec vectors
    words = []
    data = []
    sif_data = []
    for line in open(wordvec, encoding='utf-8'):
        line = line.strip().split()
        words.append(line[0])
        data.append([float(i) for i in line[1:]])
    for line in open(sif):
        line = list(map(float, line.strip().split()))
        sif_data.append(line)
    voc, pairs = readVocs(corpus, corpus_name, words, word_f)
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filterPairs(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    sentences = [pair[2].split() for pair in pairs]
    word_counts = Counter(itertools.chain(*sentences))
    word_s2s = []
    total_words = sum([k[1] for k in word_counts.most_common()])
    temp = 0
    for k, v in word_counts.most_common()[:20000]:
        word_s2s.append(k)
        temp += v
    print ('oov rate: {}'.format(1-temp/total_words))
    voc_s2s = Voc('s2s', word_s2s) 
    
    print("Counted words:", voc.n_words)
    # create definition bow
    for pair in pairs:
        definition = pair[2].split(' ')
        bow = {}
        count = torch.zeros(DEC_MAX_LENGTH )
        index = torch.ones(DEC_MAX_LENGTH) * UNK_token
        for d in definition:
            if d in voc_s2s.word2index:
                ind = voc_s2s.word2index[d]
                if ind in bow:
                    bow[ind] += 1
                else:
                    bow[ind] = 1
        for i, (k, v) in enumerate(bow.items()):
            index[i] = k
            count[i] = v
        pair.append(torch.stack((index, count)))

    directory = os.path.join(save_dir, 'training_data', corpus_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save(voc, os.path.join(directory, '{!s}.tar'.format('voc')))
    torch.save(voc_s2s, os.path.join(directory, '{!s}.tar'.format('voc_s2s')))
    torch.save(pairs, os.path.join(directory, '{!s}.tar'.format('pairs')))
    torch.save(data, os.path.join(directory, '{!s}.tar'.format('pretrained')))
    torch.save(sif_data, os.path.join(directory, '{!s}.tar'.format('sif_pretrained')))
    return voc, voc_s2s, pairs, data, sif_data

def loadPrepareData(corpus, wordvec, target_words, sif):
    corpus_name = corpus.split('/')[-1].split('.')[0]
    try:
        print("Start loading training data ...")
        voc = torch.load(os.path.join(save_dir, 'training_data', corpus_name, 'voc.tar'))
        voc_s2s = torch.load(os.path.join(save_dir, 'training_data', corpus_name, 'voc_s2s.tar'))
        pairs = torch.load(os.path.join(save_dir, 'training_data', corpus_name, 'pairs.tar'))
        pretrained = torch.load(os.path.join(save_dir, 'training_data', corpus_name, 'pretrained.tar'))
        sif_pretrained = torch.load(os.path.join(save_dir, 'training_data', corpus_name, 'sif_pretrained.tar'))
    except FileNotFoundError:
        print("Saved data not found, start preparing training data ...")
        voc, voc_s2s, pairs, pretrained, sif_pretrained = prepareData(corpus, corpus_name, wordvec, target_words, sif)
    return voc, voc_s2s, pairs, pretrained, sif_pretrained
