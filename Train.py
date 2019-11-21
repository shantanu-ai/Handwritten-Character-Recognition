import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.tensorboard import SummaryWriter

from HWCRUtils import HWCRUtils

torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True)


class Train_Manager:
    def train_data_set(self, train_set, network, run, epochs):
        print("training starts now")
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
            actual_correct = 0

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
                total_correct += HWCRUtils.get_num_correct(preds, labels)
                actual_correct += labels.shape[0]

            # tensor board tracking
            # tb.add_scalar("Loss", total_loss, epoch)
            # tb.add_scalar("Number Correct", total_correct, epoch)
            # tb.add_scalar("Accuracy", total_correct / len(train_set), epoch)

            # for name, weight in network.named_parameters():
            #     tb.add_histogram(name, weight, epoch)
            #     tb.add_histogram(f'{name}.grad', weight.grad, epoch)
            prcent_correct = (total_correct/ actual_correct) * 100
            print("epoch: {0}, total_correct: {1}, actual_correct: {2}, % correct: {3}, loss: {4}".format(epoch,
                                                                                                          total_correct,
                                                                                                          actual_correct,
                                                                                                          prcent_correct,
                                                                                                          total_loss))

        # tb.close()
        return {
            "network": network
        }
