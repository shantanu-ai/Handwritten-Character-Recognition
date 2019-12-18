import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True)


class Network(nn.Module):
    """
    This class creates the Convolution neural network without batch normalization and dropout.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=5)

        self.conv2 = nn.Conv2d(in_channels=128, out_channels=192, kernel_size=3)

        self.conv3 = nn.Conv2d(in_channels=192, out_channels=256, kernel_size=3)

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3)

        self.fc1 = nn.Linear(in_features=128 * 4 * 4, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=512)
        self.out = nn.Linear(in_features=512, out_features=9)

    def forward(self, t):
        # 1st conv layer
        if torch.cuda.is_available():
            t = t.float().cuda()
        else:
            t = t.float()

        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # 2nd conv layer
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # 3rd conv layer
        t = self.conv3(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # 4th conv layer
        t = self.conv4(t)
        t = F.relu(t)

        # 5th layer
        t = self.fc1(t.reshape(-1, 128 * 4 * 4))
        t = F.relu(t)

        t = self.fc2(t)
        t = F.relu(t)

        # output layer
        t = self.out(t)
        return t
