import torch
from torch.utils.data import TensorDataset, DataLoader
from constants import *
import numpy as np
import itertools
from collections import Counter
import pickle
import os

class Voc:
    def __init__(self, name):
        self.name = name

        if name == 's2s':
            self.word2index = {
                PAD: PAD_IDX,
                UNK: UNK_IDX,
                BOS: BOS_IDX,
                EOS: EOS_IDX
            }
            self.index2word = {v: k for k, v in self.word2index.items()}
            self.n_words = len(self.word2index)
        else:
            self.word2index = {}
            self.index2word = {}
            self.embedding = {}
            self.n_words = 0

    def add_word(self, word, vec=None):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            if self.name == 'w2v':
                self.embedding[word] = vec
            self.n_words += 1


def load_pretrain(wordvec):
    voc = Voc(name='w2v')
    with open(wordvec, 'r') as f:
        for idx, line in enumerate(f):
            word, vec = line.strip().split(' ', 1)
            word = word.lower()
            vec = np.fromstring(vec, sep=' ')
            if len(vec) != 300: continue
            voc.add_word(word, vec)          
    return voc


def load_sif(sif):
    sif_emb = []
    
    with open(sif, 'r') as f:
        for line in f:
            line = list(map(float, line.strip().split()))
            sif_emb.append(line)
    
    return sif_emb


def prepare_data(corpus, voc_w2v, sif_emb):
    
    def_sents = []
    trg_embs, ctx_embs, lengths = [], [], []
    with open(corpus, 'r') as f:
        for i, line in enumerate(f):
            w, _, defin = line.split(';')
            w = w.strip()
            sent = defin.strip().split()
            if len(sent) < DEC_MAX_LENGTH and w in voc_w2v.word2index:
                def_sents.append(sent)
                trg_embs.append(voc_w2v.embedding[w])
                ctx_embs.append(sif_emb[i])
                lengths.append(len(sent)+1) # +1 for EOS

    print("Trimmed to %s sentences" % len(def_sents))

    # get the most common words in corpus and build voc_dec
    voc_dec = Voc(name='s2s')
    count_dict = Counter(itertools.chain(*def_sents))
    for word, cnt in count_dict.most_common()[:VOC_DEC_NUM]:
        voc_dec.add_word(word)

    # string -> word idx
    def_ids = []
    for sent in def_sents:
        num_pad = DEC_MAX_LENGTH - len(sent) -1
        def_ids.append([voc_dec.word2index[w] if w in voc_dec.word2index else UNK_IDX for w in sent]+\
                        [EOS_IDX] + [PAD_IDX]*num_pad)

    assert len(def_ids) == len(trg_embs) == len(ctx_embs) == len(lengths)
    assert voc_dec.n_words == VOC_DEC_NUM + 4

    return voc_dec, trg_embs, ctx_embs, def_ids, lengths


def prepare_test(corpus, voc_w2v, sif_emb, unseen=False):
    if unseen:
        import gensim
        from gensim.models import KeyedVectors
        from gensim.models import Word2Vec
        PRETRAIN = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

    trg_words, def_sents, ctx_sents = [], [], []
    trg_embs, ctx_embs = [], []
    with open(corpus, 'r') as f:
        for i, line in enumerate(f):
            w, ctx, defin = line.split(';')
            w = w.strip()
            defin = defin.strip()
            ctx = ctx.strip()
            if unseen:
                if w not in voc_w2v.word2index and w in PRETRAIN.wv.vocab:
                    trg_words.append(w)
                    def_sents.append(defin)
                    ctx_sents.append(ctx)
                    trg_embs.append(PRETRAIN[w])
                    ctx_embs.append(sif_emb[i])
            else:
                if w in voc_w2v.word2index:
                    trg_words.append(w)
                    def_sents.append(defin)
                    ctx_sents.append(ctx)
                    trg_embs.append(voc_w2v.embedding[w])
                    ctx_embs.append(sif_emb[i])

    return trg_embs, ctx_embs, trg_words, def_sents, ctx_sents


def prepare_wic(words_file, ans_file, voc_w2v, sif_emb):
    trg_embs, ctx_embs, answers, unk_ids = [], [], [], []

    ans_list = open(ans_file, 'r').readlines()
    with open(words_file, 'r') as f:
        for i, line in enumerate(f):
            word = line.strip()
            if word in voc_w2v.word2index:
                answers.append(ans_list[i].strip())
                w_emb = voc_w2v.embedding[word]
                trg_embs.append(w_emb)
                trg_embs.append(w_emb)
                ctx_embs.append(sif_emb[i*2])
                ctx_embs.append(sif_emb[i*2+1])
            else:
                unk_ids.append(i)
            next(f)
    
    print('[{} /{}] {}'.format(len(unk_ids), len(ans_list), \
                "questions are unkown since the target word embedding is not in W2V !"))
    assert len(trg_embs) == len(ctx_embs) == 2*len(answers)

    return trg_embs, ctx_embs, answers


def loadTrainData(args):
    try:
        print("Start loading training data ...")
        with open(os.path.join(args.save_dir, 'trg_embs'), 'rb') as f:
            trg_embs = pickle.load(f)
        with open(os.path.join(args.save_dir, 'ctx_embs'), 'rb') as f:
            ctx_embs = pickle.load(f)
        with open(os.path.join(args.save_dir, 'def_ids'), 'rb') as f:
            def_ids = pickle.load(f)
        with open(os.path.join(args.save_dir, 'lengths'), 'rb') as f:
            lengths = pickle.load(f)

    except FileNotFoundError:
        print("Saved data not found, start preparing training data ...")
        voc_w2v = load_pretrain(args.w2v_file)
        sif_emb = load_sif(args.sif_file)
        voc_dec, trg_embs, ctx_embs, def_ids, lengths = prepare_data(args.corpus, voc_w2v, sif_emb)

        voc_w2v = torch.save(voc_w2v, os.path.join(args.save_dir, 'voc_w2v.tar'))
        voc_dec = torch.save(voc_dec, os.path.join(args.save_dir, 'voc_dec.tar'))

        with open(os.path.join(args.save_dir, 'trg_embs'), 'wb') as f:
            pickle.dump(trg_embs, f)
        with open(os.path.join(args.save_dir, 'ctx_embs'), 'wb') as f:
            pickle.dump(ctx_embs, f)
        with open(os.path.join(args.save_dir, 'def_ids'), 'wb') as f:
            pickle.dump(def_ids, f)
        with open(os.path.join(args.save_dir, 'lengths'), 'wb') as f:
            pickle.dump(lengths, f)

    train_data = TensorDataset(torch.FloatTensor(trg_embs), torch.FloatTensor(ctx_embs), torch.LongTensor(def_ids), torch.LongTensor(lengths))
    dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)

    return dataloader


def loadTestData(args):
    voc_w2v = torch.load(os.path.join(args.save_dir, 'voc_w2v.tar'))
    voc_dec = torch.load(os.path.join(args.save_dir, 'voc_dec.tar'))
    sif_emb = load_sif(args.sif_file)
    trg_embs, ctx_embs, trg_words, def_sents, ctx_sents = prepare_test(args.corpus, voc_w2v, sif_emb)
    
    return voc_dec, [trg_embs, ctx_embs, trg_words, def_sents, ctx_sents]


def loadWicData(args):
    voc_w2v = torch.load(os.path.join(args.save_dir, 'voc_w2v.tar'))
    sif_emb = load_sif(args.sif_file)
    trg_embs, ctx_embs, answers = prepare_wic(args.wic_words_file, args.wic_ans_file, voc_w2v, sif_emb)

    return [trg_embs, ctx_embs, answers]