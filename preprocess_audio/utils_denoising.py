import os
from scipy.io import wavfile


def read_file(file_path, sample_rate=None, trim=False):
    # Reads in a wav file and returns it as an np.float32 array in the range [-1,1]
    file_sr, data = wavfile.read(file_path)
    if data.dtype == np.int16:
        data = np.float32(data) / np.iinfo(np.int16).max
    elif data.dtype != np.float32:
        raise OSError('Encounted unexpected dtype: {data.dtype}')
    if sample_rate is not None and sample_rate != file_sr:
        if len(data) > 0:
            data = librosa.core.resample(data, file_sr, sample_rate, res_type='kaiser_fast')
        file_sr = sample_rate
    if trim and len(data) > 1:
        data = librosa.effects.trim(data, top_db=40)[0]
    return data, file_sr


def write_file(data, file_path, sample_rate=44100):
    # Writes a wav file to disk stored as int16
    if data.dtype == np.int16:
        int_data = data
    elif data.dtype == np.float32:
        int_data = np.int16(data * np.iinfo(np.int16).max)
    else:
        raise OSError(f'Input datatype {data.dtype} not supported, use np.float32')
    wavfile.write(file_path, sample_rate, int_data)