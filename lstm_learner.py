import torch
from torch import nn
from    torch.nn import functional as F
import  numpy as np

# baseline model here
class baseline_model(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super(baseline_model, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
                
        # Embedding parameters
        w_embedding = nn.Parameter(torch.ones(vocab_size, embedding_dim))
        torch.nn.init.kaiming_normal_(w_embedding)
        
        # LSTM parameters
        h_0 = nn.Parameter(torch.Tensor(embedding_dim, hidden_dim * 4))
        W = nn.Parameter(torch.Tensor(embedding_dim, hidden_dim * 4))
        U = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim * 4))
        bias = nn.Parameter(torch.zeros(hidden_dim * 4))
        torch.nn.init.kaiming_normal_(W)
        torch.nn.init.kaiming_normal_(U)
        # torch.nn.init.kaiming_normal_(bias)
        
        # Linear parameters
        w_linear = nn.Parameter(torch.ones(hidden_dim, vocab_size))
        bias_linear = nn.Parameter(torch.zeros(vocab_size))
        torch.nn.init.kaiming_normal_(w_linear)
        
        # this dict contains all tensors needed to be optimized
        self.vars = nn.ParameterList([w_embedding, W, U, bias, w_linear, bias_linear])

    def forward(self, sentence, vars=None):
        if vars is None:
            vars = self.vars
        
        x = F.embedding(sentence, vars[0])
        bs, seq_sz, _ = x.size()
        
        hidden_seq = []
        h_t, c_t = (torch.zeros(bs, self.hidden_dim).to(x.device), 
                    torch.zeros(bs, self.hidden_dim).to(x.device))
        
        W, U, bias = vars[1], vars[2], vars[3]
        HS = self.hidden_dim
        for t in range(seq_sz):
            x_t = x[:, t, :]
            # batch the computations into a single matrix multiplication
            gates = x_t @ W + h_t @ U + bias
            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :HS]), # input
                torch.sigmoid(gates[:, HS:HS*2]), # forget
                torch.tanh(gates[:, HS*2:HS*3]),
                torch.sigmoid(gates[:, HS*3:]), # output
            )
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        
        w_linear, bias_linear = vars[4], vars[5]
        prob = F.linear(hidden_seq, w_linear, bias_linear)
        
#         embeds = self.word_embeddings(sentence)
#         lstm_out, _ = self.lstm(embeds) #.view(len(sentence), 1, -1))
#         prob = self.hidden2prob(lstm_out.view(-1, lstm_out.size()[-1]))
        return prob
    
    def zero_grad(self, vars=None):
        """

        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        return self.vars