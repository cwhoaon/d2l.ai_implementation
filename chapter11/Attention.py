import torch
import torch.nn as nn
import torch.nn.functional as F

def masked_softmax(X, valid_len):
    #X:3d(batch_size, num_queries, num_keys), valid_len: 1d or 2d
    shape = X.shape
    if valid_len.dim() == 1:
        valid_len = torch.repeat_interleave(valid_len, X.shape[1])
    else:
        valid_len = valid_len.flatten()
    X = X.reshape((-1, shape[-1]))
    maxlen = X.shape[1]
    mask = torch.arange(maxlen, device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = -1e6
    
    return F.softmax(X, dim=1).reshape(shape)

class AdditiveAttention(nn.Module):
    def __init__(self, num_hiddens, dropout):
        super(AdditiveAttention, self).__init__()
        self.W_q = nn.LazyLinear(num_hiddens, bias=False)
        self.W_k = nn.LazyLinear(num_hiddens, bias=False)
        self.w_v = nn.LazyLinear(1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        scores = self.w_v(features).squeeze(3)
        self.attention_weights = masked_softmax(scores, valid_lens)
        #attention_weights: (batch_size, num_queries, num_keys)
        #values: (batch_size, num_keys, num_hiddens)
        return torch.bmm(self.dropout(self.attention_weights), values)





