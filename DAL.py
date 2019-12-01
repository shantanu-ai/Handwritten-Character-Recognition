import torch
import collections
import numpy as np
from torch import nn
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
from skimage.transform import resize


class DAL:
    def __init__(self):
        print("Reading data from pkl")
        self.data_set = None
        self.labels = None

    def read_data(self, data_path, label_path):
        try:
            self.data_set = np.load(data_path, allow_pickle=True)
            self.labels = np.load(label_path)
        except FileNotFoundError:
            raise FileNotFoundError(f'Path specified is not valid')

    def pre_process_data_set(self, img_dims):
        img_height, img_width = img_dims
        train_data = np.array([np.expand_dims(self.resize_padded(np.array(x), 64), axis=0)
                               for x in self.data_set])
        long_labels = np.array(self.labels, dtype=np.longlong)
        return train_data, long_labels

    @staticmethod
    def resize_padded(image, new_shape):
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
