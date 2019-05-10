import os
import os.path

import numpy as np
import torch.utils.data as data
# from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pdb

import deep_isp_utils as utils


class AudioDataset(data.Dataset):
    """`MSR Demosaicing <https://www.microsoft.com/en-us/download/details.aspx?id=52535>`_  Dataset.

    Args:
        root (string): Root directory of dataset where directory ``Dataset_LINEAR_with_noise`` exists.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in a pair of PIL images
            and returns a transformed version. E.g, ``transforms.RandomCrop``

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
    def read_pair_imgs(self, id):
        gt = plt.imread(os.path.join(self.dir_path, 'groundtruth', id + '.png'))[:, :, :3] - 0.5 # to check the mean ~0.17, std ~0.104   ,np.mean( np.resize( np.transpose(  plt.imread(os.path.join(self.dir_path, 'groundtruth', id + '.png'))[:, :, :3]  , (2,0,1) ) ,(3,290040) ) , axis=1 )
        gt = np.transpose(gt, (2, 0, 1))

        mosaiced = plt.imread(os.path.join(self.dir_path, 'input', id + '.png'))
        image = utils.mosaic_then_demosaic(mosaiced, 'rggb') - 0.5
        return image, gt
    """

    def __init__(self, data_dir, train=True, validation=False, validation_part=0.1, transform=None):
        assert (not (train and validation))
        self.data_dir = data_dir
        self.transform = transform
        self.train = train  # training set or test set
        self.validation = validation  # validation set

        # if not self._check_integrity():
        #    raise RuntimeError('Dataset not found or corrupted.')

        # now load the picked numpy arrays
        if self.train or self.validation:
            gt_dir = os.path.join(data_dir, 'label')
            input_dir = os.path.join(data_dir, 'train')

            self.train_list = [f[6:] for f in os.listdir(gt_dir)]  # TODO: check splitting

            self.train_data = []
            self.train_labels = []

            for f in self.train_list:
                im, gt = self.read_pair_imgs(gt_dir, input_dir, f, train=True)
                self.train_data.append(im)
                self.train_labels.append(gt)

            self.train_data, self.val_data, self.train_labels, self.val_labels = train_test_split(
                                                                                    self.train_data,
                                                                                    self.train_labels,
                                                                                    test_size=validation_part,
                                                                                    random_state=32)

        else: # in order to view images plt.imshow(gt.transpose(1, 2, 0)) ,plt.savefig('rotor.png')
            gt_dir = os.path.join(data_dir, 'test_dir_sounds')
            input_dir = os.path.join(data_dir, 'test_combined')

            self.test_data = []
            self.test_labels = []

            self.test_filenames = [f[len('test_sounds_'):] for f in os.listdir(gt_dir)]  # TODO: check splitting

            for f in self.test_filenames:
                im, gt = self.read_pair_imgs(gt_dir, input_dir, f, train=False)
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