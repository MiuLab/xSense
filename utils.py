import torch
import torch.nn as nn
from torch import optim
import numpy as np
from model import DecoderRNN, SPINEModel, MaskGenerator
from constants import VOC_DEC_NUM, VOC_W2V_NUM

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
torch.manual_seed(200)

def build_model_optimizer(args):
    print('Building models ...')
    # load pretrained model
    spine = SPINEModel(nn.Embedding(VOC_W2V_NUM, 300))
    sp_ckpt = torch.load('save/model/pretrained_spine.tar')                                                                      
    spine.load_state_dict(sp_ckpt['sp'])

    # build model
    n_voc_dec = VOC_DEC_NUM + 4
    dec_emb = nn.Embedding(n_voc_dec, 300)
    decoder = DecoderRNN(dec_emb, args.hidden_size, n_voc_dec, args.n_layers, args.dropout)
    mask_generator = MaskGenerator(z_dim=1000, enc_dim=300, K=5)

    # use cuda
    dec_emb = dec_emb.to(device)
    decoder = decoder.to(device)
    spine = spine.to(device)
    mask_generator = mask_generator.to(device)

    # optimizer
    print('Building optimizers ...')
    decoder_optimizer = optim.Adam(filter(lambda p: p.requires_grad, \
                                    list(decoder.parameters())+list(mask_generator.parameters())), lr=args.lr)

    return decoder_optimizer, decoder, mask_generator, spine


def get_mask(text_len):
    max_len = torch.max(text_len).item()
    idxes = torch.arange(0, max_len, out=torch.LongTensor(max_len)).unsqueeze(0)
    idxes = idxes.to(device)
    text_mask = (idxes < text_len.unsqueeze(1)).detach() # (batch, text_len)

    return text_mask, max_len


def compute_sparsity(X):
    non_zeros = 1. * np.count_nonzero(X)
    total = X.size
    sparsity = 100. * (1 - (non_zeros)/total)
    return sparsity


def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum().item()
    s2s_loss = torch.nn.NLLLoss()
    crossEntropy = s2s_loss(inp, target)
    loss = crossEntropy.masked_select(mask).mean()
    #loss = loss.to(device)
    return loss, nTotal


def clip_parameters(model, clip):
    if clip > 0:
        for x in model.parameters():
            x.data.clamp_(-clip, clip)