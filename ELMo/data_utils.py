
# coding: utf-8

# In[1]:

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


# In[2]:

class Vocab(object):
    
    def __init__(self, filename):
        self.idx_to_word = {}
        self.word_to_idx = {}
        self.filename = filename
#         self.num_words = num_words
        self.unk_vec = None
        self.dim = None
        
        USE_CUDA = torch.cuda.is_available()
        self.device = torch.device("cuda" if USE_CUDA else "cpu")
        with open(filename) as f:
            idx = 0
            for line in f:
                line = line.split()
                self.idx_to_word[idx] = line[0]
                self.word_to_idx[line[0]] = idx
                if not self.dim:
                    self.dim = len(line[1:])
                idx += 1
        
        self.embedding_matrix = torch.zeros(len(self.idx_to_word)+2, self.dim, device=self.device)
        
        with open(filename) as f:
            idx = 1;
            for line in f:
                line = line.split()
                self.embedding_matrix[idx] = torch.tensor(list(map(float, line[1:])), device=self.device)
                idx += 1
            self.unk_vec = torch.sum(self.embedding_matrix, 0)/(len(self.idx_to_word))
            self.embedding_matrix[len(self.idx_to_word)+1] = self.unk_vec
        
    def embedding(self, input_seq):
        MAX_LEN = input_seq.size()[0]
        batch_size = input_seq.size()[1]
        embedded = torch.zeros(MAX_LEN, batch_size, self.dim, device=self.device)
        for i in range(MAX_LEN):
            for j in range(batch_size):
                embedded[i,j,:] = self.embedding_matrix[input_seq[i, j]]
        return embedded
    
    def encode(self, sentence):
        encoded = torch.zeros(len(sentence), dtype=torch.long, device=self.device)
        idx=0
        for word in sentence:
            if word in self.word_to_idx:
                encoded[idx] = self.word_to_idx[word]
            else:
                encoded[idx] = len(self.word_to_idx)+1
            idx += 1
        
        return encoded
    
    def decode(self, sentence):
        decoded = torch.zeros(*sentence.size(), device=self.device)
        idx = 0
        for word in sentence:
            decoded[idx] = self.idx_to_word[word]
        
        return decoded
    


# In[3]:

def batcher(list_sentence, MAX_LEN, batch_size, test_ratio=0.3):
    
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")
    total = len(list_sentence)
    train_len = total*(1-test_ratio)
    test_len = total - train_len
    train_sentence = list_sentence[:int(train_len)]
    test_sentence = list_sentence[int(train_len):]
    train = []
    test = []
    idx = int(0)
    
    while True:
        pos = int(idx)
        if pos == train_len:
            break;
        if (idx + batch_size <= train_len):
            next_pos = int(idx+batch_size)
        else:
            next_pos = train_len
        t_batch = torch.zeros(MAX_LEN, int(next_pos - pos), dtype=torch.long, device=device)
        tmp_batch = train_sentence[int(pos):int(next_pos)]
        target = torch.zeros(int(next_pos - pos), dtype=torch.long, device=device)
        length = []
        mask = torch.zeros(MAX_LEN, int(next_pos-pos), dtype=torch.long, device=device)
        for batch_n in range(int(next_pos - pos)):
            l = MAX_LEN if (len(tmp_batch[batch_n])-1) > MAX_LEN else (len(tmp_batch[batch_n])-1)
            length.append(l)
            target[batch_n] = tmp_batch[batch_n][-1]
            t_batch[:l, batch_n] = tmp_batch[batch_n][:l]
            mask[:l, batch_n] = torch.ones(l)
        train.append((t_batch, target, mask, length))
        idx = next_pos
        
    idx = int(0)
    
    while True:
        pos = idx
        if pos == test_len:
            break;
        if (idx + batch_size <= test_len):
            next_pos = idx+batch_size
        else:
            next_pos = test_len
        t_batch = torch.zeros(MAX_LEN, int(next_pos - pos), dtype=torch.long, device=device)
        tmp_batch = test_sentence[int(pos):int(next_pos)]
        target = torch.zeros(int(next_pos - pos), dtype=torch.long, device=device)
        length = []
        mask = torch.zeros(MAX_LEN, int(next_pos-pos), dtype=torch.long, device=device)
        for batch_n in range(int(next_pos - pos)):
            l = MAX_LEN if (len(tmp_batch[batch_n])-1) > MAX_LEN else (len(tmp_batch[batch_n])-1)
            length.append(l)
            target[batch_n] = tmp_batch[batch_n][-1]
            t_batch[:l, batch_n] = tmp_batch[batch_n][:l]
            mask[:l, batch_n] = torch.ones(l)
        test.append((t_batch, target, mask, length))
        idx = next_pos
        
    return train, test


# In[5]:

def data_loader(data_file, vocab, print_every=20):
    import io
    import re
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")
    X = []
    with io.open(data_file, encoding='utf-8', errors='ignore') as f:
        idx = 0
        for line in f:
            line = line.lower()
            line = re.sub(r"[()\"#/@;:<>{}`+=~|.!?,]", "", line)
            new_line = line.split()
            new_line = [word.strip() for word in new_line if word != '']
            tokenized_line = vocab.encode(new_line[:-1])
            X.append(torch.cat((tokenized_line, torch.tensor([float(new_line[-1])], dtype=torch.long, device=device)), dim=0))
            idx += 1
            if idx%print_every == 0:
                print("words, sentiment =", new_line[:-1], new_line[-1])
                print("X, Y =", X[-1][:-1], X[-1][-1])
    from random import shuffle
    shuffle(X)
    return X


# In[ ]:

def make_confusion_matrix(true, pred):
    K = len(np.unique(true)) # Number of classes 
    result = np.zeros((K, K))

    for i in range(len(true)):
        result[true[i]][pred[i]] += 1

    return result

embedding_file = 'glove.6B.50d.txt'
vocab = Vocab(embedding_file)
