#code mainly base on https://medium.com/datadriveninvestor/music-generation-using-deep-learning-85010fb982e2


import keras
from keras.callbacks import CSVLogger

############PACKAGES##################
import numpy as np
import os
import json
import sys
import pandas as pd


###########Custom Packages############
from models import MusicModel

import tensorflow as tf
print(tf.test.gpu_device_name())
# See https://www.tensorflow.org/tutorials/using_gpu#allowing_gpu_memory_growth
config = tf.ConfigProto()
config.gpu_options.allow_growth = True




def read_batches(all_chars, n_vocab, batch_size=16, n_gram=64):
    length = all_chars.shape[0]
    batch_chars = int(length / batch_size) #155222/16 = 9701
    
    for start in range(0, batch_chars - n_gram, n_gram):  #(0, 9637, n_gram)  #it denotes number of batches. It runs everytime when
        #new batch is created. We have a total of 151 batches.
        X = np.zeros((batch_size, n_gram))    #(16, n_gram)
        Y = np.zeros((batch_size, n_gram, n_vocab))   #(16, n_gram, 87)
        for batch_index in range(0, 16):  #it denotes each row in a batch.  
            for i in range(0, n_gram):  #it denotes each column in a batch. Each column represents each character means 
                #each time-step character in a sequence.
                X[batch_index, i] = all_chars[batch_index * batch_chars + start + i]
                Y[batch_index, i, all_chars[batch_index * batch_chars + start + i + 1]] = 1 #here we have added '1' because the
                #correct label will be the next character in the sequence. So, the next character will be denoted by
                #all_chars[batch_index * batch_chars + start + i + 1]
        yield X, Y

def get_data_info(in_path, save_path=None):
    #mapping character to index
    vocab_map = {}
    chars = list(set(data))
    vocab_map['char2idx'] = {ch: i for (i, ch) in enumerate(chars)}
    vocab_map['idx2char'] = {i: ch for (i, ch) in enumerate(chars)}
    print("Number of unique characters in our whole tunes database = {}".format(len(vocab_map['char2idx']))) #87
    
    if save_path:
        with open(save_path, mode = "w") as f:
            json.dump(vocab_map, f)
        
    n_vocab = len(vocab_map['char2idx'])
    return vocab_map
    
def train_model(model_type, data, vocab_map, n_gram=64, model_folder='./model', batch_size=16, epochs = 80):

    n_vocab = len(vocab_map['char2idx'])
    
    all_characters = np.asarray([vocab_map['char2idx'][c] for c in data], dtype = np.int32)
    print("Total number of characters = "+str(all_characters.shape[0])) #155222
    
    epoch_number, loss, accuracy = [], [], []
    
    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch+1, epochs))
        final_epoch_loss, final_epoch_accuracy = 0, 0
        epoch_number.append(epoch+1)
        
        for i, (x, y) in enumerate(read_batches(all_characters, n_vocab)):
            final_epoch_loss, final_epoch_accuracy = model.train_on_batch(x, y) #check documentation of train_on_batch here: https://keras.io/models/sequential/
            print("Batch: {}, Loss: {}, Accuracy: {}".format(i+1, final_epoch_loss, final_epoch_accuracy))
            #here, above we are reading the batches one-by-one and train our model on each batch one-by-one.
        loss.append(final_epoch_loss)
        accuracy.append(final_epoch_accuracy)
        
        #saving weights after every 10 epochs
        if (epoch + 1) % 5 == 0:
            model_weights_directory = model_folder+'/model_weight/'
            if not os.path.exists(model_weights_directory):
                os.makedirs(model_weights_directory)
            model.save_weights(os.path.join(model_weights_directory, "Weights_{}.h5".format(epoch+1)))
            print('Saved Weights at epoch {} to file Weights_{}.h5'.format(epoch+1, epoch+1))
    
    #creating dataframe and record all the losses and accuracies at each epoch
    log_frame = pd.DataFrame(columns = ["Epoch", "Loss", "Accuracy"])
    log_frame["Epoch"] = epoch_number
    log_frame["Loss"] = loss
    log_frame["Accuracy"] = accuracy
    log_frame.to_csv(model_folder+"/log.csv", index = False)
    
    

    
    return model

if __name__ == "__main__":
    BATCH_SIZE = 16
    N_GRAM = 64
    MODEL_DIR = './music_model/'+sys.argv[1]
    DATA_PATH = './data/jig_hornpipes_cleaned.txt'
    MODEL_TYPE = 'Default'#unuse for now
    TRANSFER = False
    model_choice = 'LSTMSkipConnection'
    
    file = open(DATA_PATH, mode = 'r')
    data = file.read()
    file.close()
    
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    vocab_map = get_data_info(data, save_path=os.path.join(MODEL_DIR, 'model_dictionary.json'))
    
    n_vocab = len(vocab_map['char2idx'])
    
    if model_choice is 'LayersRNNGeneric':
        model = MusicModel(n_vocab).LayersRNNGeneric(batch_input_shape=(BATCH_SIZE, N_GRAM),
                                                    layers=['lstm', 'lstm', 'lstm'], 
                                                     layers_size=[128, 256, 128], drop_rate=0.4)
    elif model_choice is 'LSTMSkipConnection':
        model = MusicModel(n_vocab).LSTMSkipConnection(batch_input_shape=(BATCH_SIZE, N_GRAM),
                    layers=[128, 256, 128],
                                                                emb_dim=256,drop_rate=0.3)
                                                                
    
    if TRANSFER:
        model.load_weights("./music_model/04_13_19_twolayerlstm_transfer/model_weight/Weights_80.h5", by_name = True)
    
    print('\nModel choice:', model_choice)
    print('\n',model.summary())
    model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
    
    
    train_model(MODEL_TYPE, data, vocab_map,  n_gram=64, model_folder=MODEL_DIR, batch_size=16, epochs = 130)
        
        
        
