import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import os
from torch.utils.tensorboard import SummaryWriter

from HWCRUtils import HWCRUtils
from CNN import Network

from RunManager import RunManager

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
            network = self.__train_network(model_directory_path,
                                           network, train_set, run, epochs)
            print('Finished Training.')
            torch.save(network.state_dict(), model_path)
            print('Saved model parameters to disk.')

        return {
            "network": network
        }

    @staticmethod
    def __train_network(model_directory_path, network, train_set, run, epochs):
        print("training starts..")
        final_tot_correct = []
        batch_size = run.batch_size
        lr = run.lr
        shuffle = run.shuffle
        # set batch size
        data_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=1)

        # set optimizer - Adam
        optimizer = optim.Adam(network.parameters(), lr=lr)

        # initialise summary writer
        run_manager = RunManager()
        run_manager.begin_run(run, network, data_loader)

        # start training
        for epoch in range(epochs):
            run_manager.begin_epoch()

            for batch in data_loader:
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

                run_manager.track_loss(loss)
                run_manager.track_total_correct_per_epoch(preds, labels)

            run_manager.end_epoch()

        run_manager.end_run()
        path = '/Users/shantanughosh/Desktop/Shantanu_MS/Fall 19/FML/Project/Code_base/Handwritten-Character-Recognition/metrics/'

        save_file_name = path + "hwcr_cnn_lr_" + str(lr) + \
               "_batch_size_" + str(batch_size) + "shuffle_" + str(shuffle)
        run_manager.save(save_file_name)

        # print("epoch: {0}, total_correct: {1}, actual_correct: {2}, % correct: {3}, loss: {4}".format(epoch,
        #                                                                                               total_correct,
        #                                                                                               actual_correct,
        #                                                                                               prcent_correct,
        #                                                                                               total_loss))

        return network
