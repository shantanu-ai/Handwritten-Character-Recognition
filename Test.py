import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from HWCRUtils import HWCRUtils

torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True)


class Test_Manager:
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
            total_correct += HWCRUtils.get_num_correct(preds, labels)

        return {
            "network": network,
            "total_loss": total_loss,
            "total_correct": total_correct,
        }
