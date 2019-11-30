import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from CNN import Network as CNN_bn
from CNN_minusBN import Network as CNN_no_bn
from RunManager import RunManager

torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True)


class Train_Manager:
    def __init__(self):
        self.network_no_bn = CNN_no_bn()
        self.network_bn = CNN_bn()

    def train_data_set(self, train_set, run, model_directory_path, model_paths,
                       epochs):
        model_path_no_bn = model_paths[0]
        model_path_bn = model_paths[1]

        if not os.path.exists(model_directory_path):
            os.makedirs(model_directory_path)

        self.__load_model_bn(train_set, run, model_path_bn, epochs)
        self.__load_model_no_bn(train_set, run, model_path_no_bn, epochs)

        return {
            "network_bn": self.network_bn,
            "network_no_bn": self.network_no_bn
        }

    def __load_model_no_bn(self, train_set, run, model_path_no_bn, epochs):
        print("Training without batch Normalization")
        if os.path.isfile(model_path_no_bn):
            # load trained model parameters from disk
            self.network_no_bn.load_state_dict(torch.load(model_path_no_bn))
            print('Loaded model parameters from disk.')
        else:
            self.__train_network(train_set, run, epochs)
            print('Finished Training.')
            torch.save(self.network_no_bn.state_dict(), model_path_no_bn)
            print('Saved model parameters to disk.')

    def __load_model_bn(self, train_set, run, model_path_bn, epochs):
        print("Training batch Normalization")
        if os.path.isfile(model_path_bn):
            # load trained model parameters from disk
            self.network_bn.load_state_dict(torch.load(model_path_bn))
            print('Loaded model parameters from disk.')
        else:
            self.__train_network(train_set, run, epochs)
            print('Finished Training.')
            torch.save(self.network_bn.state_dict(), model_path_bn)
            print('Saved model parameters to disk.')

    def __train_network(self, train_set, run, epochs):
        print("Training starts..")
        batch_size = run.batch_size
        lr = run.lr
        shuffle = run.shuffle
        # set batch size
        data_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=1)

        # set optimizer - Adam
        optimizer_bn = optim.Adam(self.network_bn.parameters(), lr=lr)
        optimizer_no_bn = optim.Adam(self.network_no_bn.parameters(), lr=lr)

        # initialise summary writer
        run_manager_bn = RunManager()
        run_manager_no_bn = RunManager()

        run_manager_bn.begin_run(run, self.network_bn, data_loader, "With-Batch_Normalization-")
        run_manager_no_bn.begin_run(run, self.network_no_bn, data_loader, "Without-Batch_Normalization-")

        # start training
        for epoch in range(epochs):
            run_manager_bn.begin_epoch()
            run_manager_no_bn.begin_epoch()

            for batch in data_loader:
                images, labels = batch

                # forward propagation
                preds_bn = self.network_bn(images)
                preds_no_bn = self.network_no_bn(images)

                loss_bn = F.cross_entropy(preds_bn, labels)
                loss_no_bn = F.cross_entropy(preds_no_bn, labels)

                # zero out grads for every new iteration
                optimizer_bn.zero_grad()
                optimizer_no_bn.zero_grad()

                # back propagation
                loss_bn.backward()
                loss_no_bn.backward()

                # update weights
                # w = w - lr * grad_dw
                optimizer_bn.step()
                optimizer_no_bn.step()

                run_manager_bn.track_loss(loss_bn)
                run_manager_bn.track_total_correct_per_epoch(preds_bn, labels)

                run_manager_no_bn.track_loss(loss_no_bn)
                run_manager_no_bn.track_total_correct_per_epoch(preds_no_bn, labels)

            run_manager_bn.end_epoch()
            run_manager_no_bn.end_epoch()

        run_manager_bn.end_run()
        run_manager_no_bn.end_run()
        path = '/Users/shantanughosh/Desktop/Shantanu_MS/Fall 19/FML/Project/Code_base/Handwritten-Character-Recognition/metrics/'

        save_file_name_bn = path + "_with_bn_hwcr_cnn_lr_" + str(lr) + \
                         "_batch_size_" + str(batch_size) + "shuffle_" + str(shuffle)

        save_file_name_no_bn = path + "_without_bn_hwcr_cnn_lr_" + str(lr) + \
                         "_batch_size_" + str(batch_size) + "shuffle_" + str(shuffle)

        run_manager_bn.save(save_file_name_bn)
        run_manager_no_bn.save(save_file_name_no_bn)

