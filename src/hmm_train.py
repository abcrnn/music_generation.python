import numpy as np
import matplotlib.pyplot as plt
import os
from hmmlearn import hmm
import pickle
import sys
import json
from nltk import FreqDist

#########################3
    
    
############Utils#########
def save_model(hmm_model, save_name='./hmm_model.pkl'):
    with open(save_name, 'wb') as output:
        pickle.dump(hmm_model, output, pickle.HIGHEST_PROTOCOL)
def load_model(dir_path='./'):
    model = None
    with open(dir_path+'/hmm_model.pkl', 'rb') as input:
        model = pickle.load(input)
    return model


def built_model(type='MultinomialHMM', n_components=1, n_iter=10):
    if type is 'GaussianHMM':
        model = hmm.GaussianHMM(n_components=n_components, covariance_type="full", 
                                n_iter=n_iter, verbose=True, tol=0.001)
    elif type is 'MultinomialHMM':
        model = hmm.MultinomialHMM(n_components=n_components, 
                                n_iter=n_iter, verbose=True, tol=0.001)
        
    return model

def get_data_info(data, save_path=None):
    #mapping character to index
    vocab_map = {}
    chars = list(set(data))
    vocab_map['char2idx'] = {ch: i for (i, ch) in enumerate(chars)}
    vocab_map['idx2char'] = {i: ch for (i, ch) in enumerate(chars)}
    print("Number of unique characters in our whole tunes database = {}".format(len(vocab_map['char2idx']))) #87
    
    if save_path:
        with open(save_path, mode = "w+") as f:
            json.dump(vocab_map, f)
        
    n_vocab = len(vocab_map['char2idx'])
    return vocab_map

def get_train_val(data, train_frac=0.8):
    '''
    -data: np array of shape (N,) each cell in arrary
    represent the character in idx
    '''
    n_train = int(len(data) * train_frac)
    X_train, X_val = data[:n_train], data[n_train:len(data)]
    return X_train, X_val


def sampling(n_samples=100):
    X, state = model.sample(n_samples)
    seq = ''.join(vocab_map['idx2char'][c] for c in state)
        

if __name__ == "__main__":
    
    DATA_PATH = './data/jig_hornpipes_cleaned.txt'
    N_ITER = int(sys.argv[2])
    FEATURES_MULTIPLIER = 4
    MODEL_DIR = './hmm_model/' + sys.argv[1]+'_iter'+str(N_ITER) + '_featuremult'+str(FEATURES_MULTIPLIER)+ '/'
    
    
    print('FEATURES_MULTIPLIER', FEATURES_MULTIPLIER)
    print('Saving model in ', MODEL_DIR)
    #make sure path to model dir exist
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    file = open(DATA_PATH, mode = 'r')
    data = file.read()
    file.close()
    
    vocab_map = get_data_info(data, save_path=os.path.join(MODEL_DIR, 'model_dictionary.json'))
    n_vocab = len(vocab_map['char2idx'])
    
    all_characters = np.asarray([vocab_map['char2idx'][c] for c in data], dtype = np.int32)
    Xy_train, Xy_val = get_train_val(all_characters, train_frac=0.8)
    print('Init hmm model')
    model = built_model(type='MultinomialHMM', n_components=n_vocab*FEATURES_MULTIPLIER, n_iter=N_ITER)
    
    fd = FreqDist(all_characters)
    frequencies = np.fromiter((fd.freq(i) for i in range(n_vocab)), dtype=np.float64)
    emission_prob = np.stack([frequencies] * n_vocab)
    model.emissionprob_ = emission_prob
    
    
    features = np.array([Xy_train]).T
    
    print('-'*10, 'Training Hmm' , '-'*10)
    model.fit(np.atleast_2d(all_characters).T)
    
    save_path = MODEL_DIR + '/hmm_model.pkl'
    print('Saving hmm model to path', save_path)
    save_model(model, save_name=save_path)
