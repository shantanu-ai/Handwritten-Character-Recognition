import numpy as np
import torch
from torch import nn
from collections import namedtuple
from itertools import product
import sklearn.model_selection as sklearn

from DAL import DAL


class HWCRUtils:
    @staticmethod
    def numpy_load(path, allow_pickle=False):
        return np.load(path, allow_pickle=allow_pickle)

    @staticmethod
    def get_num_correct(preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()

    @staticmethod
    def spilt_data_set(data_set, label_set, split_size):
        X_train, X_test, Y_train, Y_test = \
            sklearn.train_test_split(data_set, label_set, test_size=split_size, stratify=label_set)

        return X_train, X_test, Y_train, Y_test

    @staticmethod
    def convert_to_tensor(X, Y, device):
        tensor_x = torch.stack([torch.Tensor(i) for i in X])
        tensor_y = torch.from_numpy(Y)
        processed_dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)
        return processed_dataset

    @staticmethod
    def read_dataset(data_set_path, label_set_path, image_dims):
        dal = DAL()
        dal.read_data(data_set_path, label_set_path)
        train_data_set, labels_set = dal.pre_process_data_set(image_dims)
        return train_data_set, labels_set

    @staticmethod
    def get_runs(params):
        Run = namedtuple("Run", params.keys())

        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs

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
