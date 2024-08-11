import torch
import re
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import build_vocab_from_iterator

class MTFraEng(Dataset):
    def __init__(self, path, train=True, num_steps=9, num_train=512, num_val=128):
        super(MTFraEng, self).__init__()

        self.num_steps = num_steps

        text = self._get_text(path)
        text = self._preprocess(text)
        src, tgt = self._tokenize(text)

        arrays, self.src_vocab, self.tgt_vocab = self._build_arrays(src, tgt)
        src, tgt, src_valid_len, label = arrays

        if train:
            self.src = src[:num_train]
            self.tgt = tgt[:num_train]
            self.label = label[:num_train]
            self.src_valid_len = src_valid_len[:num_train]
        else:
            self.src = src[num_train:num_train+num_val]
            self.tgt = tgt[num_train:num_train+num_val]
            self.label = label[num_train:num_train+num_val]
            self.src_valid_len = src_valid_len[num_train:num_train+num_val]

    def __getitem__(self, index):
        return self.src[index], self.tgt[index], self.src_valid_len[index], self.label[index]

    def __len__(self):
        return len(self.src)

    def _get_text(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()

    def _preprocess(self, text):
        text = re.sub('\tCC-BY(.*)', '', text)
        text = text.replace('\u202f', ' ').replace('\xa0', ' ')
        no_space = lambda char, prev_char: char in ',.!?' and prev_char != ' '
        out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
            for i, char in enumerate(text.lower())]
        return ''.join(out)

    def _tokenize(self, text, max_examples=None):
        src, tgt = [], []
        for i, line in enumerate(text.split('\n')):
            if max_examples and i > max_examples:
                break
            parts = line.split('\t')
            if len(parts) == 2:
                src.append([t for t in f'{parts[0]} <eos>'.split(' ') if t])
                tgt.append([t for t in f'{parts[1]} <eos>'.split(' ') if t])
        return src, tgt

    def _build_arrays(self, src, tgt, src_vocab=None, tgt_vocab=None):
        def _build_array(sentences, vocab, is_tgt=False):
            pad_or_trim = lambda seq, max_len: seq[:max_len] if len(seq) > max_len else seq + ['<pad>'] * (max_len - len(seq))

            sentences = [pad_or_trim(s, self.num_steps) for s in sentences]
            if is_tgt:
                sentences = [['<bos>'] + s for s in sentences]
            
            if vocab is None:
                vocab = build_vocab_from_iterator(sentences, min_freq=2, specials=["<unk>"])
                vocab.set_default_index(vocab['<unk>'])

            array = torch.tensor([vocab.lookup_indices(s) for s in sentences])
            valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
            return array, vocab, valid_len

        src_array, src_vocab, src_valid_len = _build_array(src, src_vocab)
        tgt_array, tgt_vocab, _ = _build_array(tgt, tgt_vocab, is_tgt=True)
        return (src_array, tgt_array[:,:-1], src_valid_len, tgt_array[:,1:]), src_vocab, tgt_vocab
    
    def build(self, src_sentences, tgt_sentences):
        raw_text = '\n'.join([src + '\t' + tgt for src, tgt in zip(src_sentences, tgt_sentences)])
        src, tgt = self._tokenize(raw_text)
        arrays, _, _ = self._build_arrays(src, tgt, self.src_vocab, self.tgt_vocab)
        return arrays
