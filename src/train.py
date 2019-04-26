#code mainly base on https://medium.com/datadriveninvestor/music-generation-using-deep-learning-85010fb982e2


import keras
from keras.callbacks import CSVLogger
from keras.utils.np_utils import to_categorical
from keras import optimizers
from keras.regularizers import l1_l2

############PACKAGES##################
import numpy as np
import os
import json
import sys
import pandas as pd
import time


###########Custom Packages############
from models import MusicModel

import tensorflow as tf
print(tf.test.gpu_device_name())
# See https://www.tensorflow.org/tutorials/using_gpu#allowing_gpu_memory_growth
config = tf.ConfigProto()
config.gpu_options.allow_growth = True




# class KerasBatchGenerator(object):
#     '''
#     Thanks to: https://adventuresinmachinelearning.com/keras-lstm-tutorial/
#     '''
#     def __init__(self, data, num_steps, batch_size, n_vocab, skip_step=5):
#         self.data = data
#         self.num_steps = num_steps
#         self.batch_size = batch_size
#         self.n_vocab = n_vocab
#         # this will track the progress of the batches sequentially through the
#         # data set - once the data reaches the end of the data set it will reset
#         # back to zero
#         self.current_idx = 0
#         # skip_step is the number of words which will be skipped before the next
#         # batch is skimmed from the data set
#         self.skip_step = skip_step
#         self.total_length = data.shape[0]
#         # self.batch_chars = int(length / batch_size) #155222/16 = 9701
        
#     def generate(self):

#         # while True:
#         for start in range(0, batch_chars - n_gram, n_gram):  #(0, 9637, n_gram)  #it denotes number of batches. It runs everytime when
#             x = np.zeros((self.batch_size, self.num_steps))
#             y = np.zeros((self.batch_size, self.num_steps, self.n_vocab))
#             for i in range(self.batch_size):
#                 if self.current_idx + self.num_steps >= len(self.data):
#                     # reset the index back to the start of the data set
#                     self.current_idx = 0
#                 x[i, :] = self.data[self.current_idx:self.current_idx + self.num_steps]
#                 temp_y = self.data[self.current_idx + 1:self.current_idx + self.num_steps + 1]
#                 # convert all of temp_y into a one hot representation
#                 y[i, :, :] = to_categorical(temp_y, num_classes=self.n_vocab)
#                 self.current_idx += self.skip_step
#                 yield x, y
            


# def read_batches(all_chars, n_vocab, batch_size=200, n_gram=64):
#     length = all_chars.shape[0]
#     batch_chars = int(length / batch_size) #155222/16 = 9701
    
#     for start in range(0, batch_chars - n_gram, n_gram):  #(0, 9637, n_gram)  #it denotes number of batches. It runs everytime when
#         #new batch is created. We have a total of 151 batches.
#         X = np.zeros((batch_size, n_gram))    #(16, n_gram)
#         Y = np.zeros((batch_size, n_gram, n_vocab))   #(16, n_gram, 87)
#         for batch_index in range(0, batch_size):  #it denotes each row in a batch.  
#             for i in range(0, n_gram):  #it denotes each column in a batch. Each column represents each character means 
#                 #each time-step character in a sequence.
#                 X[batch_index, i] = all_chars[batch_index * batch_chars + start + i]
#                 Y[batch_index, i, all_chars[batch_index * batch_chars + start + i + 1]] = 1 #here we have added '1' because the
#                 #correct label will be the next character in the sequence. So, the next character will be denoted by
#                 #all_chars[batch_index * batch_chars + start + i + 1]
#         yield X, Y
        
def read_batches(all_chars, n_vocab, batch_size=200, n_gram=64):
    length = all_chars.shape[0]
    batch_chars = int(length / batch_size) #155222/16 = 9701
    
    for start in range(0, batch_chars - n_gram, n_gram):  #(0, 9637, n_gram)  #it denotes number of batches. It runs everytime when
        #new batch is created. We have a total of 151 batches.
        X = np.zeros((batch_size, n_gram))    #(16, n_gram)
        Y = np.zeros((batch_size, n_gram, n_vocab))   #(16, n_gram, 87)
        for batch_index in range(0, batch_size):  #it denotes each row in a batch.  
            idx = batch_index * batch_chars + start
            X[batch_index, :] = all_chars[idx:idx+n_gram]
            Y[batch_index, :, :] = to_categorical(all_chars[idx+1:idx+n_gram+1], num_classes=n_vocab)
        yield X, Y

def get_data_info(data, save_path=None):
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

def get_train_test_val(data, frac=[0.8,0.1, 0.1]):#define train and test. cuz the rest if for validation
    '''
    -data: np array of shape (N,) each cell in arrary
    represent the character in idx
    '''
    n_train = int(len(data) * frac[0])
    n_test = int(len(data) * frac[1])
    
    train, test, val = data[:n_train], data[n_train:n_train+n_test], data[n_train+n_test:len(data)]
    return {
        'train': train,
        'test' : test, 
        'val' : val
    }
    
    
    
def train_model(model_type, data, vocab_map, n_gram=64, 
                model_folder='./model', batch_size=16, epochs = 80,
                save_weight_every=2):

    n_vocab = len(vocab_map['char2idx'])
    
    all_characters = np.asarray([vocab_map['char2idx'][c] for c in data], dtype = np.int32)
    data_split = get_train_test_val(all_characters, frac=[0.8,0.1, 0.1])
    Xy_train, Xy_val, Xy_test = data_split['train'], data_split['val'], data_split['test']
    
    print("Total number of training characters = "+str(Xy_train.shape[0])) #155222
    print("Total number of validation chars = "+str(Xy_val.shape[0]))
    print("Total number of Test chars = "+str(Xy_test.shape[0]))
    
    
    bookeeping = {}
    
    bookeeping['epoch_number'], bookeeping['train_loss'], bookeeping['train_accuracy'] = [], [], []
    bookeeping['val_loss'], bookeeping['val_accuracy'] = [], []
    bookeeping['test_loss'], bookeeping['test_accuracy'] = [], []
    
    # train_dataloader = KerasBatchGenerator(Xy_train, num_steps=n_gram, 
    #                                        batch_size=batch_size,
    #                                        n_vocab=n_vocab, skip_step=n_gram)
    # val_dataloader = KerasBatchGenerator(Xy_val, num_steps=n_gram,
    #                                      batch_size=batch_size, 
    #                                      n_vocab=n_vocab, skip_step=n_gram)
    

    
    with open(model_folder+"/log.csv", 'w+') as file_handler:
        for k, v in bookeeping.items():
            file_handler.write(str(k) + ',')
        file_handler.write('\n')
        file_handler.flush()
        for epoch in range(epochs):
            tic = time.time()
            print("\nEpoch {}/{}".format(epoch+1, epochs))
            print('-' * 15)
            final_epoch_loss, final_epoch_accuracy = 0., 0.
            final_epoch_val_loss, final_epoch_val_accuracy = 0., 0.
            bookeeping['epoch_number'].append(epoch+1)

            val_dataloader = read_batches(Xy_val, n_vocab, batch_size=batch_size, n_gram=n_gram)
            train_dataloader = read_batches(Xy_train, n_vocab, batch_size=batch_size, n_gram=n_gram)
            test_dataloader = read_batches(Xy_test, n_vocab, batch_size=batch_size, n_gram=n_gram)

            for i, (x, y) in enumerate(train_dataloader):
                final_epoch_loss, final_epoch_accuracy = model.train_on_batch(x, y) #check documentation of train_on_batch here: https://keras.io/models/sequential/
                print("Batch: {}, Loss: {}, Accuracy: {}".format(i+1, final_epoch_loss, final_epoch_accuracy))
                #here, above we are reading the batches one-by-one and train our model on each batch one-by-one.
            #https://stackoverflow.com/questions/43882796/when-does-keras-reset-an-lstm-state
            model.reset_states()#restart the state. start feeding in new seq
            toc = time.time()
            print("Finished in ", toc-tic)
            #eval on validation set
            print('\nEvaluating on validation')
            for i, (x, y) in enumerate(val_dataloader):
                final_epoch_val_loss, final_epoch_val_accuracy = model.test_on_batch(x, y)
                print("Batch: {}, Loss: {}, Accuracy: {}".format(i+1, final_epoch_val_loss,
                                                                 final_epoch_val_accuracy))

            #eval on test set
            print('\nEvaluating on test...')
            for i, (x, y) in enumerate(test_dataloader):
                final_epoch_test_loss, final_epoch_test_accuracy = model.test_on_batch(x, y)
            print('Finished epochs', epoch)
            
            bookeeping['val_loss'].append(final_epoch_val_loss)
            bookeeping['val_accuracy'].append(final_epoch_val_accuracy)

            bookeeping['train_loss'].append(final_epoch_loss)
            bookeeping['train_accuracy'].append(final_epoch_accuracy)
    
            bookeeping['test_loss'].append(final_epoch_test_loss)
            bookeeping['test_accuracy'].append(final_epoch_test_accuracy)

            #llog to csv
            # with open(model_folder+"/log.csv", 'w') as file_handler:
            for k, v in bookeeping.items():
                file_handler.write(str(v[-1]) + ',')
            file_handler.write('\n')
            file_handler.flush()

            #saving weights after every 10 epochs
            if ((epoch + 1) % save_weight_every) == 0:
                model_weights_directory = model_folder+'/model_weight/'
                if not os.path.exists(model_weights_directory):
                    os.makedirs(model_weights_directory)
                model.save_weights(os.path.join(model_weights_directory, "Weights_{}.h5".format(epoch+1)))
                print('Saved Weights at epoch {} to file Weights_{}.h5'.format(epoch+1, epoch+1))
            print()
    #creating dataframe and record all the losses and accuracies at each epoch
    log_frame = pd.DataFrame(columns = ["Epoch", "train_loss", "train_accuracy", 'val_loss', 'val_accuracy'])
    log_frame["Epoch"] = epoch_number
    log_frame["train_loss"] = loss
    log_frame["train_accuracy"] = accuracy
    log_frame["val_loss"] = val_loss
    log_frame["val_accuracy"] = val_accuracy
    log_frame.to_csv(model_folder+"/log_summary.csv", index = False)
    
    

    
    return model

if __name__ == "__main__":
    BATCH_SIZE = 16
    N_GRAM = 30
    MODEL_DIR = './music_model/'+sys.argv[1]
    # DATA_PATH = './data/classical_music_midi/processed_clean_abc_small.txt'
        DATA_PATH = './data/jazz_music_midi/processed_clean_abc_small.txt'
    # DATA_PATH = './data/jig_hornpipes_cleaned.txt'
    # DATA_PATH = './data/notthing_ham_full_cleaned.txt'
    MODEL_TYPE = 'Default'#unuse for now
    TRANSFER = False
    model_choice = 'LSTMSkipConnection'
    save_weight_every = 2
    DROP_RATE = 0.
    DROP_EMB = 0.05
    VISUALIZE_MODEL = False
    
    file = open(DATA_PATH, mode = 'r')
    data = file.read()
    file.close()
    
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    vocab_map = get_data_info(data, save_path=os.path.join(MODEL_DIR, 'model_dictionary.json'))
    
    n_vocab = len(vocab_map['char2idx'])
    
    if model_choice is 'LayersRNNGeneric':
        regularizer = l1_l2(l1=0.01, l2=0.01)
        print('regularizer\n', regularizer.__dict__)
        
        model = MusicModel(n_vocab, phase='train').LayersRNNGeneric(batch_input_shape=(BATCH_SIZE, N_GRAM),
                                                    layers=['lstm'], 
                                                     emb_dim=512,
                                                     layers_size=[256], regularizer=regularizer, 
                                                    drop_rate=DROP_RATE)#0.15
    elif model_choice is 'BidirectionalLayersRNNGeneric':
        model = MusicModel(n_vocab, phase='train').BidirectionalLayersRNNGeneric(batch_input_shape=(BATCH_SIZE, N_GRAM),
                                                    layers=['lstm', 'lstm'], 
                                                     emb_dim=256,
                                                     layers_size=[256, 128], drop_rate=DROP_RATE)#0.15
    elif model_choice is 'LSTMSkipConnection':
        
        regularizer = l1_l2(l1=1e-2, l2=0.)
        print('regularizer\n', regularizer.__dict__)
        model = MusicModel(n_vocab).LSTMSkipConnection(batch_input_shape=(BATCH_SIZE, N_GRAM),
                                    layers=[128, 256, 128],   
                                    emb_dim=256,drop_emb=DROP_EMB,
                                    drop_rate=DROP_RATE,regularizer=regularizer)#0.35
    elif model_choice is 'LSTMSkipConnectionDropBetween':
        model = MusicModel(n_vocab).LSTMSkipConnectionDropBetween(batch_input_shape=(BATCH_SIZE, N_GRAM),
                    layers=[1024], 
                    emb_dim=256, drop_rate=DROP_RATE)#0.25
    elif model_choice is 'ResidualLSTM':
        model = MusicModel(n_vocab, phase='train').ResidualLSTM(batch_input_shape=(BATCH_SIZE, N_GRAM),
                                                rnn_width=128,
                                                rnn_depth=6,
                                                emb_dim=256,
                                                drop_rate=DROP_RATE)
                                                                
    
    if TRANSFER:
        weight_path = "./music_model/04_21_19_LayersGeneric_transfer_256_128_classical_cleaned_small_batchsize20_window30/model_weight/Weights_60.h5"
        print("transfered weight from", weight_path)
        model.load_weights(weight_path, by_name = True)
    
    #vislualize model
    # from keras.utils import plot_model
    # plot_model(model, to_file=os.path.join(MODEL_DIR, 'model_architecture.png'),show_shapes=True)
    
    print('\nModel choice:', model_choice)
    print('Drop Rate:', DROP_RATE)
    print('DROP_EMB:', DROP_EMB)
    print('\n',model.summary())
    optimizer = optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False)
    print('Optimizern\n', optimizer.__dict__)
    print('model dict\n', model.__dict__)
    
    model.compile(loss = "categorical_crossentropy", optimizer = optimizer, metrics = ["accuracy"])
    
    print('Training on ', DATA_PATH)
    
    train_model(MODEL_TYPE, data, vocab_map,  n_gram=N_GRAM, model_folder=MODEL_DIR, 
                batch_size=BATCH_SIZE, epochs = 1000, 
                save_weight_every=save_weight_every)
        
        
        
