import torch
import collections
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection as sklearn

from torch.utils.tensorboard import SummaryWriter
from skimage.transform import resize
from DAL_Utils import DAL_Utils


class DAL:
    def __init__(self):
        print("Reading data from pkl")
        self.data_set = None
        self.labels = None

    def read_data(self, data_path, label_path):
        try:
            self.data_set = DAL_Utils.numpy_load(data_path, allow_pickle=True)
            self.labels = DAL_Utils.numpy_load(label_path)
        except FileNotFoundError:
            raise FileNotFoundError(f'Path specified is not valid')

    def pre_process_data_set(self):
        train_data = np.array([np.expand_dims(resize(np.array(x), (50, 50)), axis=0)
                               for x in self.data_set])
        long_labels = np.array(self.labels, dtype=np.long)
        return train_data, long_labels

    @staticmethod
    def spilt_data_set(data_set, label_set, split_size):
        X_train_set, X_val_set, y_train_set, y_val_set = \
            sklearn.train_test_split(data_set, label_set, test_size=split_size, stratify=label_set)

        return X_train_set, X_val_set, y_train_set, y_val_set

    @staticmethod
    def convert_to_tensor(data_set, label_set):
        tensor_x = torch.stack([torch.Tensor(i) for i in data_set])
        tensor_y = torch.from_numpy(label_set)
        processed_dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)
        return processed_dataset


if __name__ == "__main__":
    dal = DAL()
    dal.read_data("train_data.pkl", "finalLabelsTrain.npy")
    train_data_set, long_labels_set = dal.pre_process_data_set()
    X_train, X_test, y_train, y_test = DAL.spilt_data_set(train_data_set,
                                                          long_labels_set,
                                                          split_size=0.01)
    X_train, X_val, y_train, y_val = DAL.spilt_data_set(X_train,
                                                        y_train,
                                                        split_size=0.01)

    my_dataset = DAL.convert_to_tensor(X_train, y_train)

    print(collections.Counter(y_train))
    print(collections.Counter(y_test))
    print(collections.Counter(y_val))

    data_loader = torch.utils.data.DataLoader(
        my_dataset, batch_size=100, num_workers=1
    )
    images, labels = next(iter(data_loader))

    first_image = images[50]
    first_image = np.array(first_image, dtype="float")
    pixels = first_image.reshape((50, 50))
    plt.imshow(pixels, cmap="gray")
    plt.show()
