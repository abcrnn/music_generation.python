import keras
import numpy as np
from music21 import *
import os
import json
import traceback

def generate_sequence(model, vocab_map, initial_index=0, seq_length=300, return_heatmap=False):
    
    if return_heatmap: heat_map = np.zeros(shape=(2, 5+1, seq_length))# 1: idx, 2: is the probility of that index
    
    
    sequence_index = [initial_index]
    n_vocab = len(vocab_map['idx2char'])
    prev_note = '\n'
    stop_note = ['\n', '%', ']', '[', ' ']
    for i in range(seq_length):
        batch = np.zeros((1, 1))
        batch[0, 0] = sequence_index[-1]
        predicted_probs = model.predict_on_batch(batch).ravel()
        sample = np.random.choice(range(n_vocab), size = 1, p = predicted_probs)[0]
    
        if return_heatmap:
            heat_map[0,0,i] = sample
            heat_map[1,0,i] = predicted_probs[sample]
            heat_map[0,1:,i] = sorted_idx = np.argsort(predicted_probs)[-5:][::-1].astype(int)#reversed
            heat_map[1,1:,i] = predicted_probs[sorted_idx]

        prev_note = vocab_map['idx2char'][str(sample)]
        sequence_index.append(sample) 
    
    str_seq = seq = ''.join(vocab_map['idx2char'][str(c)] for c in sequence_index)
    
    #postprocessing here
    if return_heatmap:
        return str_seq, heat_map
    else: return str_seq
    

def save_midi(music_abc, save_path='./midi.mid', print_trace=False):
    # print(music_abc)
    try:
        abcScore = converter.parse(music_abc, format='abc')
        mf = midi.translate.streamToMidiFile(abcScore)
        print(save_path)
        mf.open(save_path, 'wb')
        mf.write()
        mf.close()
    except Exception as err:
        print('failed parsing')
        if print_trace: traceback.print_exc()
        return False
    return True
    
    
def generate_midi_batch(batch_size=10, save_dir='./generated_midi/', func_args=None):
    '''
    compose a batch of midi and save under a files
    '''
    import random
    
    header = '''
    X: 1
    T:AbcRnn
    % abcRnn
    M:4/4
    K:K
    '''
    
    vocab_map = func_args['vocab_map']
    n_vocab = len(vocab_map['idx2char'])
    model = func_args['model']
    for idx in range(batch_size):
        initial_index = np.random.randint(0, n_vocab, size=(1,))[0]
        music_abc = generate_sequence(model, vocab_map, initial_index=initial_index, seq_length=600, return_heatmap=False)
        save_file = save_dir+'/sample'+str(idx).zfill(4)+'.mid'
        if save_midi(header+music_abc, save_file):
            pass
        else:
            print("generated fail %d' sample"%idx)

    
def main():
    from models import MusicModel
    BATCH_SIZE = 1
    SEQ_LENGTH = 1
    # MODEL_DIR = './music_model/04_20_19_GenericsLayers_256_128_drop15_classical_cleaned_strict_batchsize5000_window30/'#jazz
    MODEL_DIR = './music_model/04_21_19_ResidualLSTM_depth4_width128_drop15_classical_cleaned_small_batchsize20_window30/'
    EPOCH = 21
    EVAL_FOLDER = 'test_eval_' + str(EPOCH) + 'epoch/'
    WEIGHT_PATH = MODEL_DIR+'/model_weight/'+"Weights_{}.h5".format(EPOCH)
    DROP_RATE = 0.15
    
    with open(os.path.join(MODEL_DIR, 'model_dictionary.json')) as f:
        vocab_map = json.load(f)

    n_vocab = len(vocab_map['idx2char'])
    print('n_vocab', n_vocab)
    
    # model = MusicModel(n_vocab, phase='test').LayersRNNGeneric(batch_input_shape=(BATCH_SIZE, SEQ_LENGTH),
    #                                                     layers=['lstm','lstm'], 
    #                                                      emb_dim=512,
    #                                                      layers_size=[256, 128], drop_rate=0.25)
    model = MusicModel(n_vocab, phase='test').ResidualLSTM(batch_input_shape=(BATCH_SIZE, SEQ_LENGTH),
                                            rnn_width=128,
                                            rnn_depth=4,
                                            emb_dim=256,
                                            drop_rate=DROP_RATE)
                 
    model.load_weights(WEIGHT_PATH)
    if not os.path.exists(MODEL_DIR+EVAL_FOLDER):
        os.makedirs(MODEL_DIR+EVAL_FOLDER)
    generate_midi_batch(batch_size=30, save_dir=MODEL_DIR+EVAL_FOLDER, func_args={
        'vocab_map':vocab_map,
        'model': model
    })
                        
if __name__ == "__main__":
    main()