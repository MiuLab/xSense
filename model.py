import torch
import torch.nn as nn
import torch.nn.functional as F

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
torch.manual_seed(200)

class DecoderRNN(nn.Module):
    def __init__(self, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(DecoderRNN, self).__init__()

        # Keep for reference
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)

        self.gru = nn.GRU(hidden_size*2, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout)) 
        self.gru2 = nn.GRU(hidden_size, hidden_size, n_layers) 
        
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        nn.init.xavier_uniform_(self.concat.weight)
        nn.init.xavier_uniform_(self.out.weight)

    def forward(self, input_seq, last_hidden, last_hidden2, sense_vec):
        embedded = self.embedding(input_seq)
        embedded = torch.cat((embedded, sense_vec), dim=-1)
        rnn_output, hidden = self.gru(embedded, last_hidden)
        rnn_output, hidden2 = self.gru2(rnn_output, last_hidden2)
        output = self.out(rnn_output.squeeze(0))
        prob_output = F.log_softmax(output, dim=1)

        return prob_output, hidden, hidden2

# pretrained
class SPINEModel(torch.nn.Module):

    def __init__(self, pretrain=False):
        super(SPINEModel, self).__init__()
        
        self.inp_dim = 300
        self.hdim = 1000
        self.noise_level = 0.2
        self.getReconstructionLoss = nn.MSELoss()
        self.rho_star = 1.0 - 0.85
        self.pretrain = pretrain

        # autoencoder
        self.linear1 = nn.Linear(self.inp_dim, self.hdim)
        self.linear2 = nn.Linear(self.hdim, self.inp_dim)
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)

    def forward(self, trg_emb):
        
        batch_size = trg_emb.shape[0]
        batch_x = trg_emb + torch.randn(batch_size, self.inp_dim).to(device)*self.noise_level if self.pretrain else trg_emb
        batch_y = trg_emb
        # forward
        linear1_out = self.linear1(batch_x)
        h = linear1_out.clamp(min=0, max=1) # capped relu
        out = self.linear2(h)

        # different terms of the loss
        reconstruction_loss = self.getReconstructionLoss(out, batch_y.detach()) # reconstruction loss
        psl_loss = self._getPSLLoss(h, batch_size)                              # partial sparsity loss
        asl_loss = self._getASLLoss(h)                                          # average sparsity loss
        
        return h, self.linear1.weight.data, [reconstruction_loss, psl_loss, asl_loss]


    def _getPSLLoss(self,h, batch_size):
        return torch.sum(h*(1-h)) / (batch_size*self.hdim)

    def _getASLLoss(self, h):
        temp = torch.mean(h, dim=0) - self.rho_star
        temp = temp.clamp(min=0)
        return torch.mean(temp**2)

    def orthogonalize(self, beta=0.01):
        W = self.linear1.weight.data
        W.copy_((1 + beta) * W - beta * W.mm(W.transpose(0, 1).mm(W)))

        

class MaskGenerator(torch.nn.Module):
    def __init__(self, z_dim, enc_dim, K=5, mode='soft'):
        super(MaskGenerator, self).__init__()
        self.mode = mode
        self.K = K

        self.mapping = nn.Linear(enc_dim, enc_dim, bias=False)
        self.mapping.weight.data.copy_(torch.diag(torch.ones(enc_dim))) # initialize as identity matrix

        if mode in ['soft_cat', 'hard', 'cat']:
            self.linear = nn.Linear(enc_dim*2, enc_dim)
            nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, sp_z, sp_w, ctx_emb):
        aligned_ctx = self.mapping(ctx_emb)

        topk, indices = torch.topk(sp_z, self.K, dim=1)    # (bs, k)
        topW = sp_w[indices]                               # (bs, k, 300)
        attn = nn.functional.softmax(torch.matmul(topW, aligned_ctx.unsqueeze(2)), dim=1)       
                                                           # (k, 300) matmul (bs, 300, 1) = (bs, k, 1)
        if self.mode == 'soft':
            out = torch.sum(topW * attn, dim = 1)          # (k, 300) * (bs, k, 1) = (bs, k, 300) -> (bs, 300)
        
        elif self.mode == 'soft_cat':
            out = torch.sum(topW * attn, dim = 1)
            out = torch.cat((out, aligned_ctx), dim=1) 
            out = self.linear(out)

        elif self.mode == 'hard':
            out = topW[range(topW.shape[0]), torch.argmax(attn, dim=1).squeeze(),:]
            out = torch.cat((out, aligned_ctx), dim=1) 
            out = self.linear(out)

        elif self.mode == 'cat':                           # (k, 300) * (bs, k, 1) = (bs, k, 300) -> (bs, 300)
            sp_info = torch.sum(topW * nn.functional.softmax(topk, dim=1).unsqueeze(2), dim=1)
            out = torch.cat((sp_info, aligned_ctx), dim=1) 
            out = self.linear(out)
        
        return aligned_ctx, out, attn, torch.argmax(attn, dim=1).squeeze()
