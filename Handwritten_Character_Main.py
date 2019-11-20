import torch
from itertools import product

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from CNN import Network
from CNN import Train
from CNN import RunBuilder

from DAL import DAL

from collections import OrderedDict

from torch.utils.tensorboard import SummaryWriter


def train_and_test():
    epochs = 10
    parameters = OrderedDict(
        lr=[0.01],
        batch_size=[100]
    )

    network = Network()

    dal = DAL()
    dal.read_data("train_data.pkl", "finalLabelsTrain.npy")
    train_set = dal.pre_process_data_set()
    train = Train()
    ret = train.train_data_set(train_set, network,
                               RunBuilder.get_runs(parameters)[0],
                               epochs)
    network = ret['network']

    ret = train.test_data_set(train_set, network, RunBuilder.get_runs(parameters)[0])
    print(f"total loss test: {ret['total_loss']}")
    print(f"correctly predicted: {ret['total_correct']}")
    # print(f"actual correct: {train_set.targets.numpy().shape[0]}")
    # print(f"% correct: {ret['total_correct']}/{train_set.targets.numpy().shape[0]}")


def exec_main():
    # train_with_multiple_params
    train_and_test()


if __name__ == '__main__':
    exec_main()
