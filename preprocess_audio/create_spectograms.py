import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from utils_denoising import read_file


def log_mel_spec_tfm_dir(src_dir, list_src_files, dst_dir):
    for file_name in list_src_files:
        file_path = src_dir + '\\' + file_name
        print(f'Converting {file_name} to Melspectogram')
        log_mel_spec_tfm(file_path, dst_dir)


def log_mel_spec_tfm(src_file_path, dst_dir):
    x, sample_rate = read_file(src_file_path)

    n_fft = 1024  # FFT window size
    hop_length = 256  # number of samples between successive frames
    n_mels = 40
    fmin = 20
    fmax = sample_rate / 2

    # we can also use librosa.core.stft for STFT instead of mel spectogram.
    mel_spec_power = librosa.feature.melspectrogram(x, sr=sample_rate, n_fft=n_fft,
                                                    hop_length=hop_length,
                                                    n_mels=n_mels, power=2.0,
                                                    fmin=fmin, fmax=fmax)
    mel_spec_db = librosa.power_to_db(mel_spec_power, ref=np.max)
    dst_fname = src_file_path.split('\\')[-1][:-4]
    dst_fname = dst_dir + '\\' + dst_fname + '.jpg'  # need to split src_file_path by /
    plt.imsave(dst_fname, mel_spec_db)

DATA = os.getcwd()

# these folders must be in place
NSYNTH_AUDIO = DATA+'\\data\\nsynth_audio'
TRAIN_AUDIO_PATH = NSYNTH_AUDIO+'\\train'
VALID_AUDIO_PATH = NSYNTH_AUDIO+'\\valid'
print(TRAIN_AUDIO_PATH)
# these folders will be created
NSYNTH_IMAGES = DATA+'\\data\\nsynth_images'
TRAIN_IMAGE_PATH = NSYNTH_IMAGES+'\\train'
VALID_IMAGE_PATH = NSYNTH_IMAGES+'\\valid'

train_acoustic_fnames = [f for f in os.listdir(TRAIN_AUDIO_PATH)]
valid_acoustic_fnames = [f for f in os.listdir(VALID_AUDIO_PATH)]

log_mel_spec_tfm_dir(TRAIN_AUDIO_PATH, train_acoustic_fnames, TRAIN_IMAGE_PATH)
log_mel_spec_tfm_dir(VALID_AUDIO_PATH, valid_acoustic_fnames, VALID_IMAGE_PATH)
