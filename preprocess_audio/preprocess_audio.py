import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from sklearn import model_selection
import argparse
import random
import h5py
# conda install h5py

"""
options for augmentation:
0. random from 1 to 3
1. time stretching {0.81, 0.93, 1.07, 1.23}
2. pitch shifting light {-2, -1, 1, 2}
3. pitch shifting - larger values {-3.5, -2.5, 2.5, 3.5}
"""
def add_augmentation(sound_arr, fs = 22050, augmentation=0):
    ts_list = [0.81, 0.93, 1.07, 1.23]
    ps1_list = [-2, -1, 1, 2]
    ps2_list = [-3.5, -2.5, 2.5, 3.5]
    if augmentation == 0:
        augmentation = random.randint(1,3)
    if augmentation == 1:
        # time stretching
        ts_val = random.choice(ts_list)
        augmented_sound = librosa.effects.time_stretch(sound_arr, ts_val)
    elif augmentation == 2:
        # pitch shift light
        ps1_val = random.choice(ps1_list)
        augmented_sound = librosa.effects.pitch_shift(sound_arr, fs, n_steps=ps1_val)
    else:
        # pitch shift (larger)
        ps2_val = random.choice(ps2_list)
        augmented_sound = librosa.effects.pitch_shift(sound_arr, fs, n_steps=ps2_val)
    return augmented_sound


"""
Given two np.array that were read using librosa.load, combines both.
volume1, volume2 - amount (0 to 1) for wanted volume to combine
"""
def combine_two_wavs(wav1, wav2, volume1=1, volume2=1):
    combined = (wav1*volume1+wav2*volume2)/2
    return combined


# creates spectogram from array.
# returns (spectogram, N_CHANNELS)
def create_spectogram(src_audio, N_FFT):
    spectogram = librosa.stft(src_audio, N_FFT)
    spectogram_content = np.log1p(np.abs(spectogram[np.newaxis, :, :]))
    N_CHANNELS = spectogram_content.shape[1]
    return np.squeeze(spectogram_content, axis=0), N_CHANNELS


def create_train_test_spectograms_h5file(dir_list, sounds_train, rotors_train , N_FFT, h5_group, phase='train'):
    # create sub-groupt for input (combined sounds) and gt (ground truth - only sounds)
    input_group = h5_group.create_group('input')
    gt_group = h5_group.create_group('gt')

    sound_dir, rotors_dir = dir_list
    for sound in sounds_train:
        print(f'processing {sound}')
        sound_path = os.path.join(sound_dir, sound)
        # possible augmentation
        sound_audio, fs_s = librosa.load(sound_path)
        fs = fs_s
        spectogram_sound_label, N_CHANNELS = create_spectogram(sound_audio, N_FFT)
        for rotor in rotors_train:
            rotor_path = os.path.join(rotors_dir, rotor)
            rotor_audio, fs_r = librosa.load(rotor_path)
            if fs_s != fs_r:
                print("Error. unable to combine wavs. Don't have the same fs.")
                exit()
            # pick a random part of rotors volume
            volume_rotors = random.uniform(0.1, 0.3)
            combined_audio = combine_two_wavs(rotor_audio, sound_audio, volume1=volume_rotors)
            spectogram_combined, N_CHANNELS = create_spectogram(combined_audio, N_FFT)
            # save data
            dst_name = sound[:-4] + '_' + rotor[:-4]
            input_group.create_dataset(dst_name, data=spectogram_combined)
            gt_group.create_dataset(dst_name, data=spectogram_sound_label)
    return fs, N_CHANNELS


def save_spectogram_as_png(dst_path, spectogram):
    plt.imsave(dst_path, spectogram)


"""
example of use in command line: python preprocess_audio.py -rotors_dir '...'
"""
def parse_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', '-data_dir', help='enter data dir name', default='data')
    parser.add_argument('--rotors_dir', '-rotors_dir', help='enter rotors dir name', default='rotor')
    parser.add_argument('--sounds_dir', '-sounds_dir', help='enter sounds dir name', default='Mic8')
    parser.set_defaults(console=False)
    args = parser.parse_args()
    return args


# init args and serial
args = parse_cli_args()

# get data dir, and create sub-dirs to save the spectograms
cur_dir = os.getcwd()
par_dir = os.path.abspath(os.path.join(cur_dir, os.pardir))

data_dir = os.path.join(par_dir, args.data_dir)

rotors_dir = os.path.join(data_dir, args.rotors_dir)
sounds_dir = os.path.join(data_dir, args.sounds_dir)

# given two folders, creates train:label:test folders
sounds_list = [f for f in os.listdir(sounds_dir)]
rotors_list = [f for f in os.listdir(rotors_dir)]
# split to train:test
sounds_train, sounds_test = model_selection.train_test_split(sounds_list, train_size=0.9)
rotors_train, rotors_test = model_selection.train_test_split(rotors_list, train_size=0.9)
print(f'train sounds size: {len(sounds_train)}, train rotors size: {len(rotors_train)}')
print(f'test sounds size: {len(sounds_test)}, test rotors size: {len(rotors_test)}')

N_FFT = 1024

# save the dataset as np.arrays in h5 file
hf = h5py.File('data1.h5', 'w')
# create train and test spectograms
print('processing train files')
# create group for train in the h5 file
train = hf.create_group('train')
# create the spectograms for train
train_dirs_list = [sounds_dir, rotors_dir]
fs, N_CHANNELS = create_train_test_spectograms_h5file(train_dirs_list, sounds_train, rotors_train, N_FFT, h5_group=train, phase='train')
print('processing test files')
# create group for test in the h5 file
test = hf.create_group('test')
# create the spectograms for test
test_dirs_list = [sounds_dir, rotors_dir]
create_train_test_spectograms_h5file(test_dirs_list, sounds_test, rotors_test, N_FFT, h5_group=test, phase='test')
hf.close()
