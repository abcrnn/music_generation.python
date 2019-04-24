import sys
import os
from EasyABC import midi2abc


def midi2abc_func(fname):
    argv = [fname]
    return midi2abc.main(argv)
    
def validletter(a):
    if a == 'A' or a == 'B' or a == 'C' or a == 'D' or a == 'E' or a == 'F' or a == 'G' or a == 'X' or a == 'Z':
        return 1
    if a == 'a' or a == 'b' or a == 'c' or a == 'd' or a == 'e' or a == 'f' or a == 'g' or a == 'x' or a == 'z':
        return 1
    return 0
    
def clean_abctext(text):
    '''
    Function to clean and validate char in given abc sequence of text
    thanks to: https://github.com/sarthakagarwal18/Mathematical-Mozart
    '''
#     if l[i] == ' ' or l[i] == '~' or l[i] == '(' or l[i] == ')' or l[i] == '|' or l[i] == "\\" or l[i] == "." or l[
#         i] == "+" or l[i] == "!" or l[i] == "-" or l[i] == "[" or l[i] == "]" or l[i] == "=" or l[i] == "\n" or l[
#         i] == "{" or l[i] == "}"or l[i] == "*" or l[i] == "@"or l[i] == "0" or l[i] == "#"
        
    # invalid_c = ['~', '#', '0', '!', '=', ')', '(', 
    #              '\\', '+', '-', '*', '@', ' ', '}', '{']
    
    invalid_c = ['~', '#', '0', '!', '=', ')', '(', 
                 '\\', '+', '-', '*', '@', ' ', '}', '{',
                '.']
    processed = ""
    # for idx, char in enumerate(text):
    i = 0
    while i < len(text) - 2:#avoid stepping over
        if text[i] in invalid_c:
            i += 1
            
            continue
        if validletter(text[i]) and (text[i+1] is '#' or text[i+1] is "'"):#abc2midi supports only one notation for sharp notes(#)
            processed += '^' + text[i]
            i += 2
            continue
        if text[i] is '>':#a>b means a3/2b1/2
            processed += '3/2'
            if validletter(text[i + 1]):
                processed += text[i + 1]
                i += 2
            else:
                processed += 'G'
                i += 1
            processed += '/2'
            
            continue
        if text[i] is '<':#a<b means a1/2b3/2
            processed += '/2'
            if validletter(text[i + 1]):
                processed += text[i + 1]
                i += 2
            else:
                processed += 'G'
                i += 1
            processed += '3/2'
            
            continue
        if validletter(text[i]) or ((text[i] <= 'a' or text[i] >= 'z') and (text[i] <= 'A' or text[i] >= 'Z')):#add valid letters
            processed += text[i]
        i += 1
    return processed

        
            
def make_dset(midi_folder, save_txt_path=None, verbose=True):
    midi_namming = ['.MID', '.mid', '.midi']
    
    # if save_folder is None:
    #     save_folder = midi_folder + './cleaned_data.txt'
        
    # if not os.path.exists(save_folder):
    #     os.makedirs(save_folder)
     
    full_path = ""
    n_sucessfull_conversion = 0
    prev_n_sucessfull_conversion = 0
    with open (save_txt_path, 'w+') as file_handler:
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
                        
                        save = clean_abctext(abc_txt).split('\n')[4:]
                        
                        file_handler.write('\n\\\n' + '\n'.join(save))
                        file_handler.flush()
                        
                    except:
                        n_sucessfull_conversion -= 1
                        print('error when converting file', full_path)
            if verbose:
                print('total attempt to converted', len(files))
                print('total successful', n_sucessfull_conversion - prev_n_sucessfull_conversion)
                print('Finished processing file in dir', root,'\n') 
                      
            prev_n_sucessfull_conversion = n_sucessfull_conversion
        
    print('total Successful conversion', n_sucessfull_conversion)
                
                
                
def main():
    print(midi2abc_func('test.mid'))

def main_make_dset():
    data_folder = './data/classical_music_midi'
    # data_folder = './test'
    make_dset(midi_folder=data_folder, save_txt_path=data_folder+'/processed_clean_abc_rule_strict_right.txt')
    
text = '''
[C0A,0]<[E,0

'''
if __name__ == "__main__":
    # main_make_dset()
    
    midi2abc = midi2abc_func("./data/classical_music_midi/Classical Archives - The Greats (MIDI)/Reinecke Piano Concerto n3 2mov.mid")
    # print(midi2abc,'\n')
    print(clean_abctext(midi2abc))
    
    # print(clean_abctext("[D2A,,2D,2F,2]"))
    # main()