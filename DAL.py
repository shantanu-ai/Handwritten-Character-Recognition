import torch
import collections
import numpy as np
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
        train_data = np.array([np.expand_dims(resize(np.array(x), (img_height, img_width)), axis=0)
                               for x in self.data_set])
        long_labels = np.array(self.labels, dtype=np.long)
        return train_data, long_labels
