import numpy as np
import librosa
import h5py

# reconstraction is taken from https://github.com/vadim-v-lebedev/audio_style_tranfer/blob/master/audio_style_transfer.ipynb
def spectogram_to_wav(spectogram_content, N_CHANNELS, N_FFT, fs, dst_path=None):
    spectogram_content = np.expand_dims(spectogram_content, axis=0)
    a = np.zeros_like(spectogram_content[0])
    a[:N_CHANNELS, :] = np.exp(spectogram_content[0]) - 1

    # reconstruction
    p = 2 * np.pi * np.random.random_sample(a.shape) - np.pi
    for i in range(50):
        s = a * np.exp(1j * p)
        x = librosa.istft(s)
        p = np.angle(librosa.stft(x, N_FFT))

    if dst_path:
        librosa.output.write_wav(dst_path, x, fs)
    
    return x


"""
hf = h5py.File('data.h5', 'r')
train = hf.get('train')
input = train.get('input')

train_input_list = list(input.keys())
print(train_input_list)
specto = np.array(input.get(train_input_list[0]))
spectogram_to_wav(specto, 'wavi.wav', 513, 1024, 22050)
hf.close()
"""