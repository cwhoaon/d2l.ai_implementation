import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def masked_softmax(X, valid_lens):
    #X:3d(batch_size, num_queries, num_kvs)
    #valid_lens: (batch_size) or (batch_size, num_queries)
    if valid_lens is None:
        return F.softmax(X, dim=-1)

    shape = X.shape
    if valid_lens.dim() == 1:
        valid_lens = torch.repeat_interleave(valid_lens, shape[1])
    else:
        valid_lens = valid_lens.flatten()
    
    X = X.reshape((-1, shape[-1]))
    maxlen = X.shape[1]
    mask = torch.arange(maxlen, device=X.device)[None, :] < valid_lens[:, None]
    X[~mask] = -1e6
    
    return F.softmax(X, dim=1).reshape(shape)

class DotProductAttention(nn.Module):
    def __init__(self, dropout):
        super(DotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, queries, keys, values, valid_lens=None):
        #queries: (batch_size, num_queries, d)
        d = queries.shape[1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        #scores: (batch_size, num_queries, num_kvs)
        self.attention_weights = masked_softmax(scores, valid_lens)
        #values: (batch_size, num_kvs, num_hiddens)
        return torch.bmm(self.dropout(self.attention_weights), values)

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

class MultiHeadAttention(nn.Module):
    def __init__(self, num_hiddens, num_heads, dropout, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_k = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_v = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_o = nn.LazyLinear(num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        #qkvs: (batch_size, no. of queries or key-value pairs, num_hiddens)
        queries = self.transpose_qkv(self.W_q(queries))
        keys = self.transpose_qkv(self.W_k(keys))
        values = self.transpose_qkv(self.W_v(values))
        #qkvs: (batch_size*num_heads, no. of queries or key-value pairs, num_hiddens/num_heads)

        #valid_lens: (batch_size) or (batch_size, num_queries)
        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens, self.num_heads, dim=0)

        #output: (batch_size*num_heads, num_queries, num_hiddens/num_heads)
        output = self.attention(queries, keys, values, valid_lens)
        #output_concat: (batch_size, num_queries, num_hiddens)
        output_concat = self.transpose_output(output)

        return self.W_o(output_concat)
    
    def transpose_qkv(self, X):
        #X: (batch_size, num_qs or kv_pairs, num_heads, num_hidden/num_heads)
        X = X.reshape(X.shape[0], X.shape[1], self.num_heads, -1)
        X = X.permute(0, 2, 1, 3)
        #X: (batch_size * num_heads, num_qs or kv_pairs, num_hidden/num_heads)
        X = X.reshape(-1, X.shape[2], X.shape[3])
        return X
    
    def transpose_output(self, X):
        X = X.reshape(-1, self.num_heads, X.shape[1], X.shape[2])
        X = X.permute(0, 2, 1, 3)
        X = X.reshape(X.shape[0], X.shape[1], -1)
        return X






