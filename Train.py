import os

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from CNN import Network as CNN_bn
from CNN_Dropout import Network as CNN_dropout
from CNN_minusBN import Network as CNN_no_bn
from RunManager import RunManager

torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True)


class Train_Manager:
    def __init__(self):
        self.network_no_bn = CNN_no_bn()
        self.network_bn = CNN_bn()
        self.network_bn_dropout = CNN_dropout()

    def train_data_set(self, train_set, run, model_directory_path, model_paths, save_logistics_file_path,
                       epochs):
        model_path_no_bn = model_paths[0]
        model_path_bn = model_paths[1]
        model_path_bn_dropout = model_paths[2]

        if not os.path.exists(model_directory_path):
            os.makedirs(model_directory_path)

        print("Training with batch Normalization")
        bn_final_loss = self.__load_model(train_set, run, model_path_bn, save_logistics_file_path, epochs, "BatchNorm")

        print("Training without batch Normalization")
        no_bn_final_loss = self.__load_model(train_set, run, model_path_no_bn, save_logistics_file_path, epochs,
                                             "NoBatchNorm")

        print("Training with Dropout")
        drop_out_bn_final_loss = self.__load_model(train_set, run, model_path_bn_dropout, save_logistics_file_path,
                                                   epochs, "Dropout")

        if not (bn_final_loss is None) and not (no_bn_final_loss is None) and not (drop_out_bn_final_loss is None):
            self.plot_loss_val(bn_final_loss, no_bn_final_loss, drop_out_bn_final_loss, run)

        return {
            "network_bn": self.network_bn,
            "network_no_bn": self.network_no_bn,
            "network_bn_dropout": self.network_bn_dropout
        }

    def __load_model(self, train_set, run, model_path_no_bn, save_logistics_file_path, epochs, type):
        model = self.__getModel(type)
        loss = None
        if os.path.isfile(model_path_no_bn):
            # load trained model parameters from disk
            model.load_state_dict(torch.load(model_path_no_bn))
            print('Loaded model parameters from disk.')
        else:
            loss = self.__train_network(train_set, run, save_logistics_file_path, epochs, type)
            print('Finished Training.')
            torch.save(model.state_dict(), model_path_no_bn)
            print('Saved model parameters to disk.')

        return loss

    def __train_network(self, train_set, run, save_logistics_file_path, epochs, type):
        batch_size = run.batch_size
        lr = run.lr
        shuffle = run.shuffle

        # set batch size
        data_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=1)

        save_file_name = save_logistics_file_path + self.__get_file_name(type, shuffle, lr, batch_size)
        model = self.__getModel(type)
        tb_summary = self.__get_tb_summary_title(type)

        # set optimizer - Adam

        optimizer = optim.Adam(model.parameters(), lr=lr)

        # initialise summary writer
        run_manager = RunManager()

        run_manager.begin_run(run, model, data_loader, tb_summary)

        # start training
        for epoch in range(epochs):
            run_manager.begin_epoch()

            for batch in data_loader:
                images, labels = batch

                # forward propagation
                predictions = model(images)

                loss = F.cross_entropy(predictions, labels)

                # zero out grads for every new iteration
                optimizer.zero_grad()

                # back propagation
                loss.backward()

                # update weights
                # w = w - lr * grad_dw
                optimizer.step()

                run_manager.track_loss(loss)

            run_manager.end_epoch()

        run_manager.end_run()
        run_manager.save(save_file_name)
        return run_manager.get_final_loss_val()

    @staticmethod
    def plot_loss_val(bn_loss, no_bn_loss, dropout_loss, run):
        batch_size = run.batch_size
        lr = run.lr

        plt.title(f' Learning rate: {lr} | Batch size: {batch_size}')
        plt.plot(bn_loss, 'r', label='BatchNorm')
        plt.plot(no_bn_loss, 'g', label='Without BatchNorm')
        plt.plot(dropout_loss, 'b', label='With Prediction')
        plt.legend()
        plt.show()

    def __getModel(self, type):
        if type == "BatchNorm":
            return self.network_bn
        elif type == "NoBatchNorm":
            return self.network_no_bn
        elif type == "Dropout":
            return self.network_bn_dropout

    @staticmethod
    def __get_file_name(type, shuffle, lr, batch_size):
        if type == "BatchNorm":
            return "_with_bn_hwcr_cnn_lr_" + str(lr) + \
                   "_batch_size_" + str(batch_size) + "shuffle_" + str(shuffle)
        elif type == "NoBatchNorm":
            return "_without_bn_hwcr_cnn_lr_" + str(lr) + \
                   "_batch_size_" + str(batch_size) + "shuffle_" + str(shuffle)
        elif type == "Dropout":
            return "_with_bn_do_hwcr_cnn_lr_" + str(lr) + \
                   "_batch_size_" + str(batch_size) + "shuffle_" + str(shuffle)

    @staticmethod
    def __get_tb_summary_title(type):
        if type == "BatchNorm":
            return "With-Batch_Normalization-"
        elif type == "NoBatchNorm":
            return "Without-Batch_Normalization-"
        elif type == "Dropout":
            return "WithDropout_BN-"
