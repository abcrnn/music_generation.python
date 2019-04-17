import sys
import os
from EasyABC import midi2abc


def midi2abc_func(fname):
    argv = [fname]
    return midi2abc.main(argv)
    
def make_dset(midi_folder, save_txt_path=None, verbose=True):
    midi_namming = ['.MID', '.mid', '.midi']
    
    # if save_folder is None:
    #     save_folder = midi_folder + './cleaned_data.txt'
        
    # if not os.path.exists(save_folder):
    #     os.makedirs(save_folder)
     
    full_path = ""
    n_sucessfull_conversion = 0
    prev_n_sucessfull_conversion = 0
    with open (save_txt_path, 'w') as file_handler:
        for root, dirs, files in os.walk(midi_folder):
            if verbose:
                print('\nStart processing file in dir', root) 
            for fname in files:
                file_ext = os.path.splitext(fname)[-1]
                if file_ext in midi_namming:
                    full_path = str(os.path.join(root, fname))

                    abc_txt = ""
                    try: 
                        n_sucessfull_conversion += 1
                        abc_txt = midi2abc_func(full_path)
                        file_handler.write('\n' + abc_txt)
                        
                    except:
                        n_sucessfull_conversion -= 1
                        print('error when converting file', full_path)
            if verbose:
                print('total attempt to converted', len(files))
                print('total successful', n_sucessfull_conversion - prev_n_sucessfull_conversion)
                print('Finished processing file in dir', root,'\n') 
                      
            prev_n_sucessfull_conversion = n_sucessfull_conversion
        
    print('total Successfule conversion', n_sucessfull_conversion)
                
                
                
def main():
    print(midi2abc_func('test.mid'))

def main_make_dset():
    data_folder = './data/classical_music_midi'
    # data_folder = './test'
    make_dset(midi_folder=data_folder, save_txt_path=data_folder+'/processed.txt')
              
if __name__ == "__main__":
    main_make_dset()