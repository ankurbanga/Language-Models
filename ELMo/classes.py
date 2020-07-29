
# coding: utf-8

# In[2]:

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


# In[3]:

class biLM(nn.Module):
    '''
    initialize with
    embedding: pre-trained embedding layer
    hidden_size: size of hidden_states of biLM
    n_layers: number of layers
    dropout: dropout
    '''
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(biLM, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embedding = embedding
        USE_CUDA = torch.cuda.is_available()
        self.device = torch.device("cuda" if USE_CUDA else "cpu")
        self.drop = nn.Dropout(p=dropout)
        self.forwardLSTM = nn.LSTM(hidden_size, 
                                         hidden_size, 
                                         n_layers, 
                                         dropout=(0 if n_layers == 1 else dropout))
        self.backwardLSTM = nn.LSTM(hidden_size, 
                                         hidden_size, 
                                         n_layers, 
                                         dropout=(0 if n_layers == 1 else dropout))
        
    def forward(self, input_seq, input_lengths, initial_states=None):
        '''
        input_seq: size=(MAX_LEN, batch_size)
        input_lengths: contains length of each sentence
        initial_states: tuple of initial hidden_state of LSTM, initial cell state of LSTM
        '''
        embedded = self.embedding(input_seq)
        MAX_LEN = embedded.size()[0]
        batch_size = embedded.size()[1]
        # embedded: size=(MAX_LEN, batch_size, hidden_size)
        outputs = torch.zeros(MAX_LEN, batch_size, 2, self.hidden_size, device=self.device)
        hidden_states = torch.zeros(self.n_layers * 2, MAX_LEN, batch_size, self.hidden_size, device=self.device)
        
        if not initial_states:
            initial_states = (torch.zeros(self.n_layers, 1, self.hidden_size, device=self.device), torch.zeros(self.n_layers, 1, self.hidden_size, device=self.device))
        
        for batch_n in range(batch_size):
            b_sentence = embedded[:,batch_n, :]
            length = input_lengths[batch_n]
            
            sentence = self.drop(b_sentence[:length,:])
            hidden_forward_state, cell_forward_state = initial_states
            hidden_backward_state, cell_backward_state = initial_states
            
            for t in range(length):
                output, (hidden_forward_state, cell_forward_state) = self.forwardLSTM(sentence[t].view(1, 1, -1), (hidden_forward_state, cell_forward_state))
                outputs[t, batch_n, 0, :] = output[0, 0, :]
                hidden_states[:self.n_layers, t, batch_n, :] = hidden_forward_state[:, 0, :]
                
            for t in range(length):
                output, (hidden_backward_state, cell_backward_state) = self.backwardLSTM(sentence[length - t - 1].view(1, 1, -1), (hidden_backward_state, cell_backward_state))
                outputs[length - t - 1, batch_n, 1, :] = output[0, 0, :]
                hidden_states[self.n_layers:, length - t - 1, batch_n, :] = hidden_backward_state[:, 0, :]
                
        return outputs, hidden_states, embedded


# In[4]:

class ELMo(nn.Module):
    '''
    initialize with
    
    '''
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0, l2_coef=None, do_layer_norm=False):
        super(ELMo, self).__init__()
        USE_CUDA = torch.cuda.is_available()
        self.device = torch.device("cuda" if USE_CUDA else "cpu")
        self.hidden_size = hidden_size
        self.l2_coef = l2_coef
        self.do_layer_norm = do_layer_norm
        self.n_layers = n_layers
        self.biLM = biLM(hidden_size, embedding, n_layers, dropout)
        self.W = nn.Parameter(torch.tensor([1/(2*n_layers + 1) for i in range(2*n_layers + 1)], requires_grad=True, device=self.device))
        self.gamma = nn.Parameter(torch.ones(1, requires_grad=True, device=self.device))
        
    def do_norm(layer, mask):
        masked_layer = layer * mask
        N = torch.sum(mask) * self.hidden_size
        mean = torch.sum(masked_layer)/N
        variance = torch.sum(((masked_layer - mean) * mask) ** 2) / N
        
        return F.batch_norm(layer, mean, variance)
    
    def forward(self, input_seq, input_lengths, mask, initial_states=None):
        bilm_outputs, hidden_states, embedded = self.biLM(input_seq, input_lengths, initial_states)
        concat_hidden_with_embedding = torch.cat((embedded.unsqueeze(0), hidden_states), dim=0)
        ELMo_embedding = torch.zeros(*embedded.size(), device=self.device)
        for i in range(2*self.n_layers + 1):
            w = self.W[i]
            layer = concat_hidden_with_embedding[i]
            if self.do_layer_norm:
                layer = self.do_norm(layer, mask)
            ELMo_embedding = ELMo_embedding + w * layer
        ELMo_embedding *= self.gamma
        return ELMo_embedding, bilm_outputs


# In[ ]:



