import sys
import os
from EasyABC import midi2abc


def midi2abc_func(fname):
    argv = [fname,'--non_verbose']
    midi2abc.main(argv)
    
def make_dset(midi_folder, save_folder):
    midi_namming = ['MID', 'mid', 'midi']
    for root, dirs, files in os.walk("."):
        for fname in files:
            file_ext = os.path.splitext(filename)[-1]
            if file_ext in midi_namming:
                full_path = str(os.path.join(root, fname))
                