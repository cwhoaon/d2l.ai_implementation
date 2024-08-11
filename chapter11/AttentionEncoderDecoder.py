import torch
import torch.nn as nn
from Attention import AdditiveAttention

class AttentionEncoderDecoder(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_size, num_hiddens, num_layers, dropout):
        super(AttentionEncoderDecoder, self).__init__()
        self.encoder = Encoder(src_vocab_size, embed_size, num_hiddens, num_layers, dropout)
        self.decoder = Decoder(tgt_vocab_size, embed_size, num_hiddens, num_layers, dropout)

    def forward(self, enc_X, dec_X, enc_valid_lens):
        enc_outputs = self.encoder(enc_X)
        return self.decoder(dec_X, enc_outputs, enc_valid_lens)[0]
    
    def predict(self, src, src_valid_len, tgt_vocab, num_steps, device):
        src = src.to(device)
        src_valid_len = src_valid_len.to(device)
        enc_output, hidden_state = self.encoder(src)
        
        translations = []
        dec_inputs = torch.tensor([tgt_vocab['<bos>']]).repeat(src.shape[0]).unsqueeze(1).to(device)
        for i in range(num_steps):
            output, hidden_state = self.decoder(dec_inputs, (enc_output, hidden_state), src_valid_len)
            dec_inputs = output.argmax(dim=2)
            translations.append(dec_inputs)

        translations = torch.stack(translations).squeeze(2).t()
        return translations


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers=num_layers, dropout=dropout)

    def forward(self, X):
        #X : (batch size, num_steps)
        embs = self.embedding(X.t().type(torch.int64))
        #embs : (num_steps, batch_size, embed_size)
        outputs, state = self.rnn(embs)
        #outputs : (num_steps, batch_size, num_hidden)
        #state : (num_layers, batch_size, num_hidden)
        return outputs, state

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = AdditiveAttention(num_hiddens, dropout)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers=num_layers, dropout=dropout)
        self.fc = nn.Linear(num_hiddens, vocab_size)

    def forward(self, X, enc_state, enc_valid_lens):
        embs = self.embedding(X.t().type(torch.int64))
        #embs : (num_steps, batch_size, embed_size)
        enc_outputs, hidden_state = enc_state
        enc_outputs = enc_outputs.permute(1, 0, 2)

        outputs = []
        for x in embs:
            query = hidden_state[-1].unsqueeze(1)
            context = self.attention(query, enc_outputs, enc_outputs, enc_valid_lens)
            #x: (batch_size, embed_size), context: (batch_size, 1, num_hiddens)
            x = torch.cat((x, context.squeeze(1)), dim=1)
            #x: (batch_size, embed_size+num_hiddens)
            out, hidden_state = self.rnn(x.unsqueeze(0), hidden_state)
            outputs.append(hidden_state[-1])
        
        outputs = torch.stack(outputs)
        outputs = self.fc(outputs)
        return outputs.permute(1, 0, 2), hidden_state
