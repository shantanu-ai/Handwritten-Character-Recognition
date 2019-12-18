import numpy as np
import torch
from torch import nn


class DAL:
    """
    This class reads the dataset and does all kinds of pre-processing.
    """
    def __init__(self):
        print("Reading data from pkl")
        self.data_set = None
        self.labels = None

    def read_data(self, data_path, label_path):
        """
        Reads the data from the file specified.

        :param data_path: data set path
        :param label_path: class label path

        :return: none
        """
        try:
            self.data_set = np.load(data_path, allow_pickle=True)
            self.labels = np.load(label_path)
        except FileNotFoundError:
            raise FileNotFoundError(f'Path specified is not valid')

    def read_data_test(self, data_path):
        """
        Reads the dataset.

        :param data_path:
        :return:
        """
        try:
            self.data_set = np.load(data_path, allow_pickle=True)
            # self.data_set = self.data_set[50:250]
        except FileNotFoundError:
            raise FileNotFoundError(f'Path specified is not valid')

    def pre_process_data_set(self, img_dims):
        """
        Preprocess the images in the data set to make each image dimension as 64 X 64.

        :param img_dims: image dimension

        :return: preprocessed the data set
        """
        train_data = np.array([np.expand_dims(self.resize_padded(np.array(x), 64), axis=0)
                               for x in self.data_set])
        long_labels = np.array(self.labels, dtype=np.longlong)
        return train_data, long_labels

    def pre_process_data_set_test(self, img_dims):
        """
        Preprocess the images in the test data set to make each image dimension as 64 X 64.
        :param img_dims: image dimension

        :return: preprocessed the data set
        """
        train_data = np.array([np.expand_dims(self.resize_padded(np.array(x), 64), axis=0)
                               for x in self.data_set])
        return train_data

    @staticmethod
    def resize_padded(image, new_shape):
        """
        Resize each image of the data set to the dimension of 64 X 64 using padding.

        :param image: image in the data set
        :param new_shape: 64 X 64

        :return:  64 X 64 image
        """
        img = torch.from_numpy(image)
        delta_width = new_shape - img.shape[1]
        delta_height = new_shape - img.shape[0]
        pad_width = delta_width // 2
        pad_height = delta_height // 2
        if delta_width % 2 != 0:
            if delta_height % 2 != 0:
                pad = nn.ConstantPad2d((pad_width, pad_width + 1, pad_height, pad_height + 1), False)
            else:
                pad = nn.ConstantPad2d((pad_width, pad_width + 1, pad_height, pad_height), False)
        else:
            if delta_height % 2 != 0:
                pad = nn.ConstantPad2d((pad_width, pad_width, pad_height, pad_height + 1), False)
            else:
                pad = nn.ConstantPad2d((pad_width, pad_width, pad_height, pad_height), False)
        return pad(img).numpy()
