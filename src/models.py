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
                          Lambda,
                          Bidirectional,
                          LSTM, RNN, GRU)



from keras.layers.merge import add

# from AttentionWeightedAverage import AttentionWeightedAverage

class MusicModel(object):
    def __init__(self, n_vocab, phase='train'):
        self.n_vocab = n_vocab
        self.phase = phase
        pass
    
    def OneLayerLSTM(self, batch_input_shape=None, emb_dim=256, drop_rate=0.2):
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
    
    def TwoLayerLSTM(self, batch_input_shape=None, emb_dim=256,drop_rate=0.35):
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
    
    def BidirectionalLayersRNNGeneric(self, batch_input_shape=None, layers=['lstm', 'lstm', 'lstm'], 
                         layers_size=[128, 128, 128], emb_dim=512,
                         drop_rate=0.35):
        self.model = Sequential([Embedding(input_dim = self.n_vocab, output_dim = emb_dim, batch_input_shape= batch_input_shape)], 
                                name = "embd_1")
        
        for idx, layer in enumerate(layers):
            return_sequences = True
            if self.phase is 'test' and idx == len(layers)-1:
                return_sequences = False
            
            if layer is 'lstm':
                self.model.add(
                    Bidirectional(
                        LSTM(layers_size[idx], return_sequences = return_sequences, 
                         stateful = True, 
                         name='lstm_%d_%d'%(idx, layers_size[idx])),
                    name='bi_%d_%d'%(idx, layers_size[idx]))
                )
            elif layer is 'gru':
                self.model.add(GRU(layers_size[idx], return_sequences = return_sequences, stateful = True))
            elif layer is 'rnn':
                self.model.add(RNN(layers_size[idx], return_sequences = return_sequences, stateful = True))
            else:
                raise ValueError ("Unknown layer name", layer)

            self.model.add(Dropout(drop_rate, name='drop_%d'%idx))
            
        if self.phase is 'train': 
            self.model.add(
                TimeDistributed(
                    Dense(self.n_vocab, name='dense_%s'%('_'.join([str(s) for s in layers_size]))),
                    name='timedistributed%s'%('_'.join([str(s) for s in layers_size]))
                )
                
            )
        elif self.phase is 'test': self.model.add(Dense(self.n_vocab))
        self.model.add(Activation("softmax"))
        
        return self.model
    
    def LayersRNNGeneric(self, batch_input_shape=None, layers=['lstm', 'lstm', 'lstm'], 
                         layers_size=[128, 128, 128], emb_dim=512, regularizer=None,
                         drop_rate=0.35):
        self.model = Sequential([Embedding(input_dim = self.n_vocab, output_dim = emb_dim, batch_input_shape= batch_input_shape)], 
                                name = "embd_1")
        
        for idx, layer in enumerate(layers):
            return_sequences = True
            if self.phase is 'test' and idx == len(layers)-1:
                return_sequences = False
            
            if layer is 'lstm':
                self.model.add(
                    LSTM(layers_size[idx], 
                         return_sequences = return_sequences, 
                         stateful = True, 
                         dropout=drop_rate,
                         recurrent_dropout = drop_rate,
                         kernel_regularizer = regularizer,
                         name='lstm_%d_%d'%(idx, layers_size[idx]))
                )
            elif layer is 'gru':
                self.model.add(GRU(layers_size[idx], return_sequences = return_sequences, stateful = True))
            elif layer is 'rnn':
                self.model.add(RNN(layers_size[idx], return_sequences = return_sequences, stateful = True))
            else:
                raise ValueError ("Unknown layer name", layer)
            
        if self.phase is 'train': 
            self.model.add(
                TimeDistributed(
                    Dense(self.n_vocab, name='dense_%s'%('_'.join([str(s) for s in layers_size]))),
                    name='timedistributed%s'%('_'.join([str(s) for s in layers_size]))
                    
                )
                
            )
        elif self.phase is 'test': self.model.add(Dense(self.n_vocab))
        self.model.add(Activation("softmax"))
        
        return self.model
    def LSTMSkipConnection(self, layers=[128, 128], batch_input_shape=None, emb_dim=256,drop_emb=0.15, 
                           drop_rate=0.45,
                            regularizer=None):
        '''
        #: Recurrent dropout masks (or "drops") the connections between the recurrent units; that would be the horizontal arrows in your picture.
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
        
        stateful = None
        if self.phase is 'train': #use default
            stateful = True
            recurrent_initializer='orthogonal'
            kernel_initializer='glorot_uniform'
        else: 
            stateful = False
            recurrent_initializer='random_uniform'
            kernel_initializer='random_uniform'
            
        layer_output = []
        prev_out = drop
        for idx, layer in enumerate(layers): 
            out = LSTM(layer, return_sequences = True, 
                       stateful = stateful, 
                       dropout = drop_rate, 
                       recurrent_dropout = drop_rate,
                       kernel_regularizer = regularizer,
                       implementation=1,
                       recurrent_initializer=recurrent_initializer,
                       kernel_initializer=kernel_initializer,
                       name='lstm_layer%d'%idx)(prev_out)
            layer_output.append(out)
            prev_out = out
            
            
        seq_concat = concatenate([embedded]+layer_output,
                                 name='rnn_concat')
        
        # attention = AttentionWeightedAverage(name='attention')(seq_concat)
        output_layer = TimeDistributed(Dense(self.n_vocab, name='output', activation='softmax'))(seq_concat)
        
        self.model = Model(inputs=[input_layer], outputs=[output_layer])
        
        return self.model
    

    def LSTMSkipConnectionDropBetween(self, layers=[128, 128], batch_input_shape=None, emb_dim=256,drop_rate=0.15):
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
        
        layer_output = []
        prev_out = embedded
        for idx, layer in enumerate(layers): 
            out = LSTM(layer, return_sequences = True, stateful = True,dropout=drop_rate, name='lstm_%d_%d'%(idx,layer))(prev_out)
            layer_output.append(out)
            prev_out = out
            
            
        seq_concat = concatenate([embedded]+layer_output,
                                 name='rnn_concat')
        
        # attention = AttentionWeightedAverage(name='attention')(seq_concat)
        output_layer = TimeDistributed(Dense(self.n_vocab, name='output', activation='softmax'))(seq_concat)
        
        self.model = Model(inputs=[input_layer], outputs=[output_layer])
        
        return self.model
        


    def ResidualLSTM(self, batch_input_shape, rnn_width, rnn_depth, emb_dim=256, drop_rate=0.10):
        """
        The intermediate LSTM layers return sequences, while the last returns a single element.
        The input is also a sequence. In order to match the shape of input and output of the LSTM
        to sum them we can do it only for all layers but the last.
        Thanks to: https://gist.github.com/bzamecnik/8ed16e361a0a6e80e2a4a259222f101e
        """
        input_layer = Input(batch_shape=batch_input_shape, name='input')
        
        embedded = Embedding(input_dim = self.n_vocab, 
                               output_dim = emb_dim, name='embedding_1')(input_layer)
        
        output = embedded
        
        if self.phase is 'train':
            for i in range(rnn_depth):
                return_sequences = True 
                x_rnn = LSTM(rnn_width, recurrent_dropout=drop_rate, 
                             dropout=drop_rate, 
                             return_sequences=return_sequences, name='lstm_%d'%i)(output)
                if return_sequences:
                    # Intermediate layers return sequences, input is also a sequence.
                    if i > 0 or embedded.shape[-1] == rnn_width:
                        output = add([output, x_rnn])
                        if i == rnn_depth-1: #last layer
                            output_layer = TimeDistributed(Dense(self.n_vocab, name='output'))(output)
                    else:
                        # Note that the input size and RNN output has to match, due to the sum operation.
                        # If we want different rnn_width, we'd have to perform the sum from layer 2 on.
                        output = x_rnn
        elif self.phase is 'test':
            for i in range(rnn_depth):
                return_sequences = i < rnn_depth - 1
                x_rnn = LSTM(rnn_width, recurrent_dropout=drop_rate, 
                             dropout=drop_rate, 
                             return_sequences=return_sequences)(output)
                if i < rnn_depth-1:
                    # Intermediate layers return sequences, input is also a sequence.
                    if i > 0 or embedded.shape[-1] == rnn_width:
                        output = add([output, x_rnn])
                    else:
                        # Note that the input size and RNN output has to match, due to the sum operation.
                        # If we want different rnn_width, we'd have to perform the sum from layer 2 on.
                        output = x_rnn
                else:
                    # Last layer does not return sequences, just the last element
                    # so we select only the last element of the previous output.
                    def slice_last(output):
                        return output[..., -1, :]
                    output = add([Lambda(slice_last)(output), x_rnn])
                    output_layer = Dense(self.n_vocab, name='output')(output)
                    


                    
        output_layer = Activation("softmax")(output_layer)
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
        