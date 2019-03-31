import torch
import numpy as np
import pickle
import os
import visualize_online
from model import SPINEModel

def dump_vectors(X, outfile, words):
    print ("shape", X.shape) # (n_voc, 1000)
    
    with open(outfile, 'w') as fw:
        for i in range(len(X)):
            fw.write(words[i] + " ")
            fw.write(" ".join(str(x) for x in X[i])+"\n")


def find_top_closest_words(words, vecs):
    if not os.path.exists('word2vec_emb_out'): # dump spine embeddings
        dump_batch = 2048
        ret = []
        for i in range(0, len(vecs), dump_batch):
            trg_emb = torch.FloatTensor(vecs[i: i+dump_batch]).cuda()
            sp_z, _, _ = spine(trg_emb)
            ret.extend(sp_z.cpu().data.numpy())
        dump_vectors(np.array(ret), 'word2vec_emb_out', words)

    if not os.path.exists('vectors.pkl') or not os.path.exists('top_k_words.pkl'):
        visualize_online.load_vectors('word2vec_emb_out')

    with open('vectors.pkl','rb') as f:
        vectors = pickle.load(f)
    with open('top_k_words.pkl','rb') as f:
        top_k_words = pickle.load(f)
   
    with open('word_clusters.txt', 'w') as f:
        dump_batch = 2048
        ret = []
        for b in range(0, len(vecs), dump_batch):
            word = words[b: b+dump_batch]
            trg_emb = torch.FloatTensor(vecs[b: b+dump_batch]).cuda()

            sp_z, _, loss_terms = spine(trg_emb)
            sp_z = sp_z.detach().cpu().numpy()
            
            for (w, h) in zip(word, sp_z): # iterate through a batch
                index = np.argsort(h)[::-1][:5]
                f.write("----------------------------------------------------------------------------------------------------------\n")
                f.write('target word is: {}\n'.format(w))
                for i in index:
                    cluster = [tup[1] for tup in top_k_words[i]]
                    f.write('dim {}: {}\n'.format(i, ', '.join(cluster)))


if __name__ == '__main__':
    spine = SPINEModel(pretrain=False)
    sp_ckpt = torch.load('save/model/pretrained_spine.tar')
    spine.load_state_dict(sp_ckpt['spine'])
#    spine = SPINEModel(torch.nn.Embedding(29194, 300))
#    sp_ckpt = torch.load('pretrained_spine.tar')                                                                      
#    spine.load_state_dict(sp_ckpt['sp'])
    spine = spine.cuda()

    voc_w2v = torch.load(os.path.join('save', 'voc_w2v.tar'))
    words = list(voc_w2v.embedding.keys())
    vecs = list(voc_w2v.embedding.values())

    find_top_closest_words(words, vecs)
