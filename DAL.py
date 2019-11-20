import torch
import numpy as np
import matplotlib.pyplot as plt

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
        tensor_x = torch.stack([torch.Tensor(i) for i in train_data])
        tensor_y = torch.Tensor(self.labels)
        processed_dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)
        return processed_dataset


if __name__ == "__main__":
    dal = DAL()
    dal.read_data("train_data.pkl", "finalLabelsTrain.npy")
    my_dataset = dal.pre_process_data_set()
    data_loader = torch.utils.data.DataLoader(
        my_dataset, batch_size=100, num_workers=1
    )
    images, labels = next(iter(data_loader))

    first_image = images[50]
    first_image = np.array(first_image, dtype="float")
    pixels = first_image.reshape((50, 50))
    plt.imshow(pixels, cmap="gray")
    plt.show()
