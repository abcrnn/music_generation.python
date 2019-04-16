import os
import json
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.models import Model
from keras.layers import (Dropout, TimeDistributed, 
                          Dense, Activation, Embedding,
                          Input, concatenate,
                          LSTM, RNN, GRU)
# from AttentionWeightedAverage import AttentionWeightedAverage

class MusicModel(object):
    def __init__(self, n_vocab ):
        self.n_vocab = n_vocab
        pass
    
    def OneLayerLSTM(self, batch_input_shape=None, emb_dim=512, drop_rate=0.2):
        '''
        batch_input_shape = (batch_size, seq_len)
        '''
        self.model = Sequential()
        
        self.model.add(Embedding(input_dim = self.n_vocab, output_dim = emb_dim, batch_input_shape= batch_input_shape)) 

        self.model.add(LSTM(256, return_sequences = True, stateful = True))
        self.model.add(Dropout(drop_rate))
        self.model.add(TimeDistributed(Dense(self.n_vocab)))
        self.model.add(Activation("softmax"))
        return self.model
    
    def TwoLayerLSTM(self, batch_input_shape=None, emb_dim=512,drop_rate=0.2):
        '''
        batch_input_shape = (batch_size, seq_len)
        '''
        self.model = Sequential([
            Embedding(input_dim = self.n_vocab, output_dim = emb_dim, batch_input_shape= batch_input_shape),
            LSTM(256, return_sequences = True, stateful = True),
            Dropout(drop_rate),
            LSTM(256, return_sequences = True, stateful = True), 
            Dropout(drop_rate),
            TimeDistributed(Dense(self.n_vocab)),
            Activation("softmax")
        ])
        return self.model
    
    def LayersRNNGeneric(self, batch_input_shape=None, layers=['lstm', 'lstm', 'lstm'], 
                         layers_size=[128, 128, 128], emb_dim=512,
                         drop_rate=0.35):
        self.model = Sequential([Embedding(input_dim = self.n_vocab, output_dim = emb_dim, batch_input_shape= batch_input_shape)])
        
        for idx, layer in enumerate(layers):
            if layer is 'lstm':
                self.model.add(LSTM(layers_size[idx], return_sequences = True, stateful = True))
            elif layer is 'gru':
                self.model.add(GRU(layers_size[idx], return_sequences = True, stateful = True))
            elif layer is 'rnn':
                self.model.add(RNN(layers_size[idx], return_sequences = True, stateful = True))
            else:
                raise ValueError ("Unknown layer name", layer)
            self.model.add(Dropout(drop_rate))
        self.model.add(TimeDistributed(Dense(self.n_vocab)))
        self.model.add(Activation("softmax"))
        
        return self.model
    def LSTMSkipConnection(self, layers=[128, 128], batch_input_shape=None, emb_dim=256,drop_rate=0.35):
        '''
        Using Tensorflow graph style
        code base inspired by deepmoji by Felbo et al
        https://arxiv.org/pdf/1708.00524.pdf
        '''
        input_layer = Input(batch_shape=batch_input_shape, name='input')
        embedded = Embedding(input_dim = self.n_vocab, 
                               output_dim = emb_dim,
                               # batch_input_shape= batch_input_shape,
                               name='embedding')(input_layer)
        drop = Dropout(drop_rate, name='drop_layer1')(embedded)
        lstm_layer1 = LSTM(128, return_sequences = True, stateful = True, name='lstm_layer1')(drop)
        lstm_layer2 = LSTM(128, return_sequences = True, stateful = True, name='lstm_layer2')(lstm_layer1)
        

        layer_output = []
        prev_out = drop
        for idx, layer in enumerate(layers): 
            out = LSTM(layer, return_sequences = True, stateful = True, name='lstm_layer%d'%idx)(prev_out)
            layer_output.append(out)
            prev_out = out
            
            
        seq_concat = concatenate([embedded]+layer_output,
                                 name='rnn_concat')
        
        # attention = AttentionWeightedAverage(name='attention')(seq_concat)
        output_layer = TimeDistributed(Dense(self.n_vocab, name='output', activation='softmax'))(seq_concat)
        
        self.model = Model(inputs=[input_layer], outputs=[output_layer])
        
        return self.model
        
        
    
    def OneLayerGru(self, batch_input_shape=None, emb_dim=512,drop_rate=0.2):
        self.model = Sequential([
            Embedding(input_dim = unique_chars, output_dim = emb_dim, batch_input_shape = batch_input_shape),
            LSTM(256, return_sequences = True, stateful = True),
            Dropout(drop_rate),
            TimeDistributed(Dense(self.n_vocab)),
            Activation("softmax")
        ])
        return self.model
    
    def OneLayerRNN(self, batch_input_shape=None, emb_dim=512,drop_rate=0.2):
        self.model = Sequential([
            Embedding(input_dim = unique_chars, output_dim = emb_dim, batch_input_shape = batch_input_shape),
            RNN(256, return_sequences = True, stateful = True),
            Dropout(drop_rate),
            TimeDistributed(Dense(self.n_vocab)),
            Activation("softmax")
        ])
        return self.model
        