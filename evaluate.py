import torch
from utils import *
from load import loadTestData
from constants import BOS_IDX, EOS_IDX, DEC_MAX_LENGTH
import visualize_online

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


def id2sent(out_ids, voc_dec):
    words = []
    for idx in out_ids:
        idx = idx.item()
        if idx == EOS_IDX:
            break
        else:
            words.append(voc_dec.index2word[idx])

    return ' '.join(words)


def evaluateTestFile(decoder, spine, mask_generator, voc_dec, test_data):
    trg_embs, ctx_embs, trg_words, def_sents, ctx_sents = test_data
    out_file = open('outfile.txt', 'w')
    bs = 1024
    max_sum = 0
    for i in range(0, len(trg_embs), bs):
        trg_emb = torch.FloatTensor(trg_embs[i: i+bs]).to(device)
        ctx_emb = torch.FloatTensor(ctx_embs[i: i+bs]).to(device)

        sp_z, sp_w, loss_terms = spine(trg_emb)
        sense_vec, attn, indices = mask_generator(sp_z, sp_w, ctx_emb)
        max_sum += torch.sum(torch.max(attn, dim=1)[0])
        sense_vec = sense_vec.unsqueeze(0)

        decoder_input = torch.LongTensor([[BOS_IDX] * len(trg_emb)]).to(device)
        decoder_hidden = trg_emb.unsqueeze(0)     
        decoder_hidden2 = sense_vec

        for j in range(DEC_MAX_LENGTH):
            decoder_output, decoder_hidden, decoder_hidden2 = decoder(
                decoder_input, decoder_hidden, decoder_hidden2, sense_vec
            )
            # decoder_output: (bs, n_voc), decoder_input: (1, bs)
            out = torch.argmax(decoder_output, dim=1)   # (bs,)
            preds = torch.cat((preds, out.unsqueeze(1)), dim=1) if j else out.unsqueeze(1)
            decoder_input = out.unsqueeze(0)
        
        # preds: (bs, max_len)
        for j in range(preds.shape[0]):
            w = trg_words[i+j]
            ctx = ctx_sents[i+j]
            truth = def_sents[i+j]
            pre = id2sent(preds[j], voc_dec)

            out_file.write('{} ; {} ; {} ; {}\n'.format(w, ctx, truth, pre))
    
    print('avg highes attn value:', max_sum.item()/len(trg_embs))
    out_file.close()


def runTest(args):
    voc_dec, test_data = loadTestData(args)
    torch.set_grad_enabled(False)
    decoder, spine, mask_generator = load_model(args)
    evaluateTestFile(decoder, spine, mask_generator, voc_dec, test_data)



