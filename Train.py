import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import os
from torch.utils.tensorboard import SummaryWriter

from HWCRUtils import HWCRUtils
from CNN import Network

torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True)


class Train_Manager:
    def train_data_set(self, train_set, run, model_directory_path, model_path,
                       epochs):
        network = Network()

        if not os.path.exists(model_directory_path):
            os.makedirs(model_directory_path)

        if os.path.isfile(model_path):
            # load trained model parameters from disk
            network.load_state_dict(torch.load(model_path))
            print('Loaded model parameters from disk.')
        else:
            network = self.__train_network(network, train_set, run, epochs)
            print('Finished Training.')
            torch.save(network.state_dict(), model_path)
            print('Saved model parameters to disk.')

        return {
            "network": network
        }

    def __train_network(self, network, train_set, run, epochs):
        print("training starts now")
        final_tot_correct = []
        batch_size = run.batch_size
        lr = run.lr

        # set batch size
        data_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                                  num_workers=1, shuffle=True)

        # set optimizer - Adam
        optimizer = optim.Adam(network.parameters(), lr=lr)

        # start training
        for epoch in range(epochs):
            total_loss = 0
            total_correct = 0
            actual_correct = 0

            for batch in data_loader:
                predictions = self.__train_per_batch(batch, network, optimizer)
                total_loss += predictions["loss"]
                total_correct += predictions["total_correct"]
                actual_correct += predictions["actual_correct"]

        percent_correct = (total_correct / actual_correct) * 100
        print("epoch: {0}, total_correct: {1}, actual_correct: {2}, % correct: {3}, loss: {4}".format(epoch,
                                                                                                      total_correct,
                                                                                                      actual_correct,
                                                                                                      percent_correct,
                                                                                                      total_loss))

        return network

    @staticmethod
    def __train_per_batch(batch, network, optimizer):
        images, labels = batch
        # forward propagation
        preds = network(images)

        loss = F.cross_entropy(preds, labels)

        # zero out grads for every new iteration
        optimizer.zero_grad()

        # back propagation
        loss.backward()

        # update weights
        # w = w - lr * grad_dw
        optimizer.step()
        return {
            "loss": loss.item(),
            "total_correct": HWCRUtils.get_num_correct(preds, labels),
            "actual_correct": labels.shape[0]
        }
