import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

from collections import namedtuple
from itertools import product

from torch.utils.tensorboard import SummaryWriter

torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True)


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        self.fc1 = nn.Linear(in_features=12 * 9 * 9, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=9)

    def forward(self, t):
        # input layer
        t = t

        # 1st conv layer
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # 2nd conv layer
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # 3rd FC1
        t = self.fc1(t.reshape(-1, 12 * 9 * 9))
        t = F.relu(t)

        # 4th FC2
        t = self.fc2(t)
        t = F.relu(t)

        # output layer
        t = self.out(t)

        return t


class Train:
    def get_num_correct(self, preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()

    def train_data_set(self, train_set, network, run, epochs):
        final_tot_correct = []
        batch_size = run.batch_size
        lr = run.lr

        # set batch size
        data_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, num_workers=1)

        # print(network.conv1.weight.grad.shape)
        # print(network.conv2.weight.grad.shape)

        # set optimizer - Adam
        optimizer = optim.Adam(network.parameters(), lr=lr)

        # initialise summary writer
        comment = f' batch_size={batch_size} lr={lr}'
        # tb = SummaryWriter(comment=comment)

        # test tensor board
        images, labels = next(iter(data_loader))
        grid = torchvision.utils.make_grid(images)
        # tb.add_image("images", grid)
        # tb.add_graph(network, images)

        # start training
        for epoch in range(epochs):
            total_loss = 0
            total_correct = 0

            for batch in data_loader:
                images, labels = batch

                # forward propagation
                preds = network(images)
                # print(preds.shape)
                # estimate loss
                # print(preds)
                # print(labels)
                #
                # print(preds.shape)
                # print(labels.shape)
                loss = F.cross_entropy(preds, labels)

                # zero out grads for every new iteration
                optimizer.zero_grad()

                # back propagation
                loss.backward()

                # update weights
                # w = w - lr * grad_dw
                optimizer.step()

                total_loss += loss.item()
                total_correct += self.get_num_correct(preds, labels)

            # tensor board tracking
            # tb.add_scalar("Loss", total_loss, epoch)
            # tb.add_scalar("Number Correct", total_correct, epoch)
            # tb.add_scalar("Accuracy", total_correct / len(train_set), epoch)

            # for name, weight in network.named_parameters():
            #     tb.add_histogram(name, weight, epoch)
            #     tb.add_histogram(f'{name}.grad', weight.grad, epoch)

            print("epoch: {0}, total_correct: {1}, loss: {2}".format(epoch, total_correct, total_loss))

        #tb.close()
        return {
            "network": network
        }

    def test_data_set(self, test_set, network, run):
        final_tot_correct = []
        batch_size = run.batch_size
        lr = run.lr

        # set batch size
        data_loader = torch.utils.data.DataLoader(
            test_set, batch_size=batch_size, num_workers=1
        )

        # print(network.conv1.weight.grad.shape)
        # print(network.conv2.weight.grad.shape)

        # set optimizer - Adam
        optimizer = optim.Adam(network.parameters(), lr=lr)

        # start training
        total_loss = 0
        total_correct = 0

        for batch in data_loader:
            images, labels = batch

            # forward propagation
            preds = network(images)

            # estimate loss
            loss = F.cross_entropy(preds, labels)

            # zero out grads for every new iteration
            optimizer.zero_grad()

            # back propagation
            loss.backward()

            # update weights
            # w = w - lr * grad_dw
            optimizer.step()

            total_loss += loss.item()
            total_correct += self.get_num_correct(preds, labels)

        return {
            "network": network,
            "total_loss": total_loss,
            "total_correct": total_correct,
        }


class RunBuilder:
    @staticmethod
    def get_runs(params):
        Run = namedtuple("Run", params.keys())

        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs
