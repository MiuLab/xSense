import torch
from utils import *
from load import loadWicData

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


def evaluateWic(spine, mask_generator, wic_data):
    trg_embs, ctx_embs, wic_ans = wic_data

    bs = 1024
    assert bs % 2 == 0, "each two pair should be considered together"
    attn_sum = 0
    num_agree, num_correct = 0, 0
    
    for i in range(0, len(trg_embs), bs):
        trg_emb = torch.FloatTensor(trg_embs[i: i+bs]).to(device)
        ctx_emb = torch.FloatTensor(ctx_embs[i: i+bs]).to(device)

        sp_z, sp_w, _ = spine(trg_emb)
        _, _, attn, topID = mask_generator(sp_z, sp_w, ctx_emb)
        attn_sum += torch.sum(torch.max(attn, dim=1)[0])

        # iterate through the batch # TODO: low efficiency
        for j in range(0, len(topID), 2):
            agree = (topID[j].item() == topID[j+1].item())
            num_agree += agree

            if (wic_ans[(i+j)//2] == "T" and agree) or (wic_ans[(i+j)//2] == "F" and not agree):
                num_correct += 1
            
    print("predicited pos : neg = {} : {}".format(num_agree, len(wic_ans)-num_agree))
    print("[Accuracy] {:.3f}".format(num_correct/len(wic_ans)))
    print('avg highest attn value: {:.3f}'.format(attn_sum.item()/len(trg_embs)))  


def runWic(args):
    wic_data = loadWicData(args)
    torch.set_grad_enabled(False)
    _, spine, mask_generator = load_model(args)
    evaluateWic(spine, mask_generator, wic_data)



