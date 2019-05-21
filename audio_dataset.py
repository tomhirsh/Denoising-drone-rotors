import os
import os.path

import numpy as np
from random import randint, uniform
import torch.utils.data as data
# from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import h5py
import deep_isp_utils as utils
from preprocess_audio.preprocess_audio import combine_two_wavs, create_spectogram
import soundfile as sf    
import librosa                                                  


class AudioDataset(data.Dataset):
    """`MSR Demosaicing <https://www.microsoft.com/en-us/download/details.aspx?id=52535>`_  Dataset.

    Args:
        root (string): Root directory of dataset where directory ``Dataset_LINEAR_with_noise`` exists.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in a pair of PIL images
            and returns a transformed version. E.g, ``transforms.RandomCrop``

    """

    """
    def read_pair_imgs(self, gt_dir, input_dir, file_name, train=True):
        if train:
          file_gt_path = os.path.join(gt_dir, 'label_' + file_name)
          file_input_path = os.path.join(input_dir, 'train_' + file_name)
        else:
          file_gt_path = os.path.join(gt_dir, 'test_sounds_' + file_name)
          file_input_path = os.path.join(input_dir, 'test_combined_' + file_name)

        gt = plt.imread(file_gt_path)[:, :, :3]
        gt = np.transpose(gt, (2, 0, 1))

        input = plt.imread(file_input_path)[:, :, :3]
        input = np.transpose(input, (2, 0, 1))

        return input, gt

    """
    def read_pair_from_h5(self, gt_group, input_group, file_name):
        gt = np.array(gt_group.get(file_name))
        gt = np.expand_dims(gt, axis=0)

        input = np.array(input_group.get(file_name))
        input = np.expand_dims(input, axis=0)

        return input, gt


    def __init__(self, data_h5_path, add_rpm=False, train=True, validation=False, validation_part=0.1, transform=None):
        assert (not (train and validation))
        self.data_h5_path = data_h5_path
        self.transform = transform
        self.train = train  # training set or test set
        self.validation = validation  # validation set

        # if not self._check_integrity():
        #    raise RuntimeError('Dataset not found or corrupted.')

        hf = h5py.File(self.data_h5_path, 'r')

        # now load the picked numpy arrays
        if self.train or self.validation:
            train_sub = hf.get('train')
            input_sub = train_sub.get('input')
            gt_sub = train_sub.get('gt')

            self.train_filenames = list(input_sub.keys())

            self.train_data = []
            self.train_labels = []

            for f in self.train_filenames:
                im, gt = self.read_pair_from_h5(gt_sub, input_sub, f)
                # add rpm as channel
                if add_rpm:
                    rpm = f.split("_")[-2]
                    rpm_channel = np.full_like(im, rpm)
                    im = np.append(im, rpm_channel, 0)
                
                self.train_data.append(im)
                self.train_labels.append(gt)

            self.train_data, self.val_data, self.train_labels, self.val_labels = train_test_split(
                                                                                    self.train_data,
                                                                                    self.train_labels,
                                                                                    test_size=validation_part,
                                                                                    random_state=32)
        else:
            test_sub = hf.get('test')
            input_sub = test_sub.get('input')
            gt_sub = test_sub.get('gt')

            self.test_filenames = list(input_sub.keys())

            self.test_data = []
            self.test_labels = []

            for f in self.test_filenames:
                im, gt = self.read_pair_from_h5(gt_sub, input_sub, f)
                # add rpm as channel
                if add_rpm:
                    rpm = f.split("_")[-2]
                    rpm_channel = np.full_like(im, rpm)
                    im = np.append(im, rpm_channel, 0)
                
                self.test_data.append(im)
                self.test_labels.append(gt)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
            filename = 0
        elif self.validation:
            img, target = self.val_data[index], self.val_labels[index]
            filename = 0
        else:
            img, target, filename = self.test_data[index], self.test_labels[index], self.test_filenames[index]

        if self.transform is not None:
            img, target = self.transform(img, target)
        return img, target, filename

    def __len__(self):
        if self.train:
            return len(self.train_data)
        elif self.validation:
            return len(self.val_data)
        else:
            return len(self.test_data)

class AudioGenDataset(data.Dataset):
    """`MSR Demosaicing <https://www.microsoft.com/en-us/download/details.aspx?id=52535>`_  Dataset.

    Args:
        root (string): Root directory of dataset where directory ``Dataset_LINEAR_with_noise`` exists.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in a pair of PIL images
            and returns a transformed version. E.g, ``transforms.RandomCrop``

    """

    """
    def read_pair_imgs(self, gt_dir, input_dir, file_name, train=True):
        if train:
          file_gt_path = os.path.join(gt_dir, 'label_' + file_name)
          file_input_path = os.path.join(input_dir, 'train_' + file_name)
        else:
          file_gt_path = os.path.join(gt_dir, 'test_sounds_' + file_name)
          file_input_path = os.path.join(input_dir, 'test_combined_' + file_name)

        gt = plt.imread(file_gt_path)[:, :, :3]
        gt = np.transpose(gt, (2, 0, 1))

        input = plt.imread(file_input_path)[:, :, :3]
        input = np.transpose(input, (2, 0, 1))

        return input, gt

    """
    # def read_pair_from_h5(self, gt_group, input_group, file_name):
    #     gt = np.array(gt_group.get(file_name))
    #     gt = np.expand_dims(gt, axis=0)

    #     input = np.array(input_group.get(file_name))
    #     input = np.expand_dims(input, axis=0)

    #     return input, gt


    def __init__(self, dataset_dir, add_rpm=False, train=True, validation=False, validation_part=0.1, transform=None, dataset_size=100, sample_length=1):
        assert (not (train and validation))
        self.dataset_dir = dataset_dir
        self.dataset_size = dataset_size
        self.rotor_dir = os.path.join(self.dataset_dir, 'rotor')
        self.transform = transform
        self.train = train  # training set or test set
        self.validation = validation  # validation set
        self.sample_length = 1

        # now load the picked numpy arrays
        if self.train or self.validation:

            self.data_dir = os.path.join(self.dataset_dir, 'train')
            self.train_filenames = os.listdir(self.data_dir)
            self.rotor_filenames = os.listdir(self.rotor_dir)

            self.train_data = []
            self.train_labels = []

            for idx in np.random.randint(0, len(self.train_filenames), self.dataset_size):
                file_path = os.path.join(self.data_dir, self.train_filenames[idx])
                file_name, ext = os.path.splitext(file_path)
                gt, sr = sf.read(file_path)

                # pick random location in file
                sample_start = randint(0, len(gt) - (sr * sample_length) - 1)
                gt = gt[sample_start: sample_start + (sr * sample_length)]
                gt = librosa.core.to_mono(np.swapaxes(gt, 0, 1))
                # pick random rotor rpm
                rotor_file_path = os.path.join(self.rotor_dir, self.rotor_filenames[randint(0, len(self.rotor_filenames)-1)])
                rotor_sound, r_sr = sf.read(rotor_file_path)

                rotor_sound = librosa.core.resample(rotor_sound, r_sr, sr)
                
                # theoretically take random sample of sample_size seconds from rotor file

                # combine sound and rotor
                volume_rotors = uniform(0.1, 0.3)
                im = combine_two_wavs(rotor_sound, gt, volume1=volume_rotors)

                N_FFT = 1024
                # convert wav to spectogram
                im, _  = create_spectogram(im, N_FFT)
                gt, _ = create_spectogram(gt, N_FFT)

                # add rpm as channel
                if add_rpm:
                    rpm = os.path.basename(rotor_file_path)
                    rpm_channel = np.full_like(im, rpm)
                    im = np.append(im, rpm_channel, 0)
                
                self.train_data.append(im)
                self.train_labels.append(gt)

            self.train_data, self.val_data, self.train_labels, self.val_labels = train_test_split(
                                                                                    self.train_data,
                                                                                    self.train_labels,
                                                                                    test_size=validation_part,
                                                                                    random_state=32)
        else:
            # test_sub = hf.get('test')
            # input_sub = test_sub.get('input')
            # gt_sub = test_sub.get('gt')

            # self.test_filenames = list(input_sub.keys())

            # self.test_data = []
            # self.test_labels = []

            # for f in self.test_filenames:
            #     im, gt = self.read_pair_from_h5(gt_sub, input_sub, f)
            #     # add rpm as channel
            #     if add_rpm:
            #         rpm = f.split("_")[-2]
            #         rpm_channel = np.full_like(im, rpm)
            #         im = np.append(im, rpm_channel, 0)
                
            #####################################################
            # self.data_dir = os.path.join(self.dataset_dir, 'train')
            # self.train_filenames = os.listdir(self.data_dir)
            # self.rotor_filenames = os.listdir(self.rotor_dir)

            # self.train_data = []
            # self.train_labels = []

            # for idx in np.random.randint(0, len(self.train_filenames), self.dataset_size):
            #     file_path = os.path.join(self.data_dir, self.train_filenames[idx])
            #     file_name, ext = os.path.splitext(file_path)
            #     gt, sr = sf.read(file_name)

            #     # pick random location in file
            #     sample_start = randint(0, len(gt) - (sr * sample_length))
            #     gt = gt[sample_start: sample_start + (sr * sample_length)]

            #     # pick random rotor rpm
            #     rotor_file_path = os.path.join(self.rotor_dir, self.rotor_filenames[randint(0, len(self.rotor_filenames)])
            #     rotor_sound = sf.read(rotor_file_path)
                
            #     # theoretically take random sample of sample_size seconds from rotor file

            #     # combine sound and rotor
            #     volume_rotors = random.uniform(0.1, 0.3)
            #     im = combine_two_wavs(rotor_sound, gt, volume1=volume_rotors)

            #     # convert wav to spectogram
            #     im, _  = create_spectogram(img, N_FFT)
            #     gt, _ = create_spectogram(gt, N_FFT)

            #     # add rpm as channel
            #     if add_rpm:
            #         rpm = os.path.basename(rotor_file_path)
            #         rpm_channel = np.full_like(im, rpm)
            #         im = np.append(im, rpm_channel, 0)
            
                self.test_data.append(im)
                self.test_labels.append(gt)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
            filename = 0
        elif self.validation:
            img, target = self.val_data[index], self.val_labels[index]
            filename = 0
        else:
            img, target, filename = self.test_data[index], self.test_labels[index], self.test_filenames[index]

        if self.transform is not None:
            img, target = self.transform(img, target)
        return img, target, filename

    def __len__(self):
        if self.train:
            return len(self.train_data)
        elif self.validation:
            return len(self.val_data)
        else:
            return len(self.test_data)
