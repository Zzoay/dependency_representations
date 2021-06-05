
import os
from abc import ABCMeta, abstractmethod
from typing import List, Tuple, Callable, Union

import numpy as np
import torch
from torch.utils.data import Dataset, random_split
import torch.nn.utils.rnn as rnn_utils


class Dependency():
    def __init__(self, id, word, tag, head, rel):
        self.id = id
        self.word = word
        self.tag = tag
        self.head = head
        self.rel = rel

    def __str__(self):
        # example:  1	上海	_	NR	NR	_	2	nn	_	_
        values = [str(self.id), self.word, "_", self.tag, "_", "_", str(self.head), self.rel, "_", "_"]
        return '\t'.join(values)

    def __repr__(self):
        return f"({self.word} {self.tag})"



class CTBDataset(Dataset):
    def __init__(self, vocab, config: dict):
        self.words, self.tags, self.heads, self.rels, self.masks, self.seq_lens = self.read_data(vocab, config["data_path"])

    def read_data(self, 
                  vocab, 
                  data_path: str,         # word, tag, head, rel, mask
                  max_len: int = None,
                  ) -> Tuple[List[torch.Tensor], ...]:
        seq_len_lst:List = []

        w_tk_lst:List = []
        t_tk_lst:List = []
        h_tk_lst:List = []
        r_tk_lst:List = []
        m_tk_lst:List = []
        for sentence in load_ctb(data_path):   # sentence: [dep, dep, ...]; dep.attrs: id, word, tag, head, rel
            seq_len = len(sentence)

            word_tokens = np.zeros(seq_len, dtype=np.int64)  # <pad> is 0 default
            tag_tokens = np.zeros(seq_len, dtype=np.int64)
            head_tokens = np.zeros(seq_len, dtype=np.int64) 
            rel_tokens = np.zeros(seq_len, dtype=np.int64)
            mask_tokens = np.zeros(seq_len, dtype=np.int64)
            for i,dep in enumerate(sentence):
                if i == seq_len:
                    break
                word = vocab.word2id.get(dep.word)
                word_tokens[i] =  word or 2  # while OOV set <unk> token
                tag_tokens[i] = (vocab.tag2id.get(dep.tag) if word else 0) or 0  # if there is no word or not tag, set 0

                head_idx = (vocab.head2id.get(dep.head) if word else 0) or 0
                if head_idx < seq_len:  # if idx in bounds, set idx into tokens
                    head_tokens[i] = head_idx

                rel_tokens[i] = (vocab.rel2id.get(dep.rel) if word else 0) or 0
                mask_tokens[i] = 1 if word else 0   # if is there a word, mask = 1, else 0 

            seq_len_lst.append(torch.tensor(seq_len))     

            w_tk_lst.append(torch.tensor(word_tokens))
            t_tk_lst.append(torch.tensor(tag_tokens))
            h_tk_lst.append(torch.tensor(head_tokens))
            r_tk_lst.append(torch.tensor(rel_tokens))
            m_tk_lst.append(torch.tensor(mask_tokens))

        return w_tk_lst, t_tk_lst, h_tk_lst, r_tk_lst, m_tk_lst, seq_len_lst
    
    def __getitem__(self, idx):
        return self.words[idx], self.tags[idx], self.heads[idx], self.rels[idx], self.masks[idx], self.seq_lens[idx]
    
    def __len__(self):
        return len(self.words)


def load_ctb(data_path: str):
    file_names:List[str] = os.listdir(data_path)
    ctb_files:List[str] = [data_path+fle for fle in file_names]

    # id, form, tag, head, rel
    sentence:List[Dependency] = []

    for ctb_file in ctb_files:
        with open(ctb_file, 'r', encoding='utf-8') as f:
            # data example: 1	上海	_	NR	NR	_	2	nn	_	_
            for line in f.readlines():
                toks = line.split()
                if len(toks) == 0:
                    yield sentence
                    sentence = []
                elif len(toks) == 10:
                    dep = Dependency(toks[0], toks[1], toks[3], toks[6], toks[7])
                    sentence.append(dep)


class Vocab():
    def __init__(self, config):
        self.min_freq = config["min_freq"]

        self.word2id = self.read_vocab(config["word_vocab_file"], isword=True)
        self.tag2id = self.read_vocab(config["tag_vocab_file"])
        self.head2id = self.read_vocab(config["head_vocab_file"])
        self.rel2id = self.read_vocab(config["rel_vocab_file"])

        self.word_size = len(self.word2id)
        self.tag_size = len(self.tag2id) + 1  # start with 1, 0 represents the <unk> or <pad> or OOV word's tag
        self.rel_size = len(self.rel2id) + 1  # start with 1, 0 represents the <unk> or <pad> or OOV word's relation

    def read_vocab(self, vocab_file: str, isword: bool = False) -> dict:
        vocab = {}
        with open(vocab_file, "r", encoding='utf-8') as f:  # TODO: encoding unify
            if isword:  # 'word' starts with <pad>, <root>, <unk>
                cnt = 0 
            else:
                cnt = 1  
            for line in f.readlines():
                word, freq = line.split()
                if isword and int(freq) < self.min_freq:  # word's freq less than min freq
                    continue
                vocab[word] = cnt
                cnt += 1
        return vocab


def train_val_split(train_dataset: Dataset, val_ratio: float, shuffle: bool = True) -> List: # Tuple[Subset] actually
    size = len(train_dataset)  # type: ignore
    val_size = int(size * val_ratio)
    train_size = size - val_size
    if shuffle:
        return random_split(train_dataset, (train_size, val_size), None)
    else:
        return [train_dataset[:train_size], train_dataset[train_size:]]


class Collator():
    def __init__(self):
        __metaclass__ = ABCMeta
    
    @abstractmethod
    def _collate_fn(self, batch):
        raise NotImplementedError

    @abstractmethod
    def __call__(self, batch):
        return self._collate_fn(batch)


class SortPadCollator(Collator):
    def __init__(self, sort_key: Callable, ignore_indics: Union[int, list] = [], reverse: bool = True):
        self.sort_key = sort_key
        self.ignore_indics = ignore_indics
        self.reverse = reverse
    
    def _collate_fn(self, batch):
        if isinstance(batch, list):
            assert self.sort_key, "if batch is a list, sort_key should be provided"  
            """
            param 'key' specifies what sort depends on.
            example: key=lambda x: x[5]; while 5 indicates index of the sequences lenght
                     key=lambda x: len(x); while sequences lenght is not provided
            """
            batch.sort(key=self.sort_key, reverse=self.reverse)  
        elif isinstance(batch, torch.Tensor):
            batch.sort(dim=-1, descending=self.reverse)

        if self.ignore_indics is None:
            self.ignore_indics = []
        elif isinstance(self.ignore_indics, int):
            self.ignore_indics = [self.ignore_indics]

        ret = []
        for i, samples in enumerate(zip(*batch)):
            if i in self.ignore_indics:
                ret.append(torch.tensor(samples))
                continue
            samples = rnn_utils.pad_sequence(samples, batch_first=True, padding_value=0)  # padding
            ret.append(samples)
        return ret

    def __call__(self, batch):
        return self._collate_fn(batch)