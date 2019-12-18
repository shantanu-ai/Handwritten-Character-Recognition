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
    """
    This class trains the cnn model.
    """
    def __init__(self):
        self.model = None

    def train_data_set(self, train_set, run, model_directory_path, model_path, save_logistics_file_path,
                       epochs, type_of_model, show_plot):
        """
        This method trains the data set.

        :param train_set: training data set
        :param run: run parameters
        :param model_directory_path: path of the directory in the disk where the model resides
        :param model_path: path of the model in the disk
        :param save_logistics_file_path: logistics path where details about model will be saved
        :param epochs:
        :param type_of_model: whether {batch normalization, no batch normalization or dropout}
        :param show_plot:

        :return: trained cnn model
        """
        model_updated = None

        if not os.path.exists(model_directory_path):
            os.makedirs(model_directory_path)

        if type_of_model == "BatchNorm":
            model = self.__getModel(type_of_model)
            print("Training with batch Normalization")
            model_updated = self.__load_model(model, train_set, run, model_path, save_logistics_file_path, epochs,
                                              type_of_model, show_plot)

        elif type_of_model == "NoBatchNorm":
            model = self.__getModel(type_of_model)
            print("Training without batch Normalization")
            model_updated = self.__load_model(model, train_set, run, model_path, save_logistics_file_path, epochs,
                                              type_of_model, show_plot)

        elif type_of_model == "Dropout":
            model = self.__getModel(type_of_model)
            print("Training with Dropout")
            model_updated = self.__load_model(model, train_set, run, model_path, save_logistics_file_path,
                                              epochs, type_of_model, show_plot)

        return model_updated

    def __load_model(self, model, train_set, run, model_path_no_bn, save_logistics_file_path, epochs,
                     type_of_model, show_plot):
        """
        Loads the model either from the disk if the model exists in the disk or trains the new model.
        and saves it in the disk
        :param model: cnn model
        :param train_set: training data set
        :param run: run parameters
        :param model_path_no_bn: path of the model in the disk
        :param save_logistics_file_path: logistics path where details about model will be saved
        :param epochs:
        :param type_of_model: whether {batch normalization, no batch normalization or dropout}
        :param show_plot:

        :return: the cnn model
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if os.path.isfile(model_path_no_bn):
            # load trained model parameters from disk
            model.load_state_dict(torch.load(model_path_no_bn, map_location=device))
            print('Loaded model parameters from disk.')
        else:
            model = self.__train_network(model, train_set, run, save_logistics_file_path, epochs,
                                         type_of_model, show_plot)
            print('Finished Training.')
            torch.save(model.state_dict(), model_path_no_bn)
            print('Saved model parameters to disk.')

        return model

    def __train_network(self, model, train_set, run, save_logistics_file_path, epochs, type_of_model, show_plot):
        """
        Trains the cnn model if the model does not exist in the disk and also saves in the disk.

        :param model: cnn model
        :param train_set: training data set
        :param run: run parameters
        :param save_logistics_file_path: logistics path where details about model will be saved
        :param epochs:
        :param type_of_model: whether {batch normalization, no batch normalization or dropout}
        :param show_plot:

        :return: trained cnn model
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("-------------------------------------------------------------------", device)
        loss_val = []
        acc_val = []
        batch_size = run.batch_size
        lr = run.lr
        shuffle = run.shuffle

        # set batch size
        data_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=1,
                                                  pin_memory=True)

        save_file_name = save_logistics_file_path + self.__get_file_name(type_of_model, shuffle, lr, batch_size)
        # model = self.__getModel(type_of_model)
        tb_summary = self.__get_tb_summary_title(type_of_model)

        # set optimizer - Adam

        optimizer = optim.Adam(model.parameters(), lr=lr)

        # initialise summary writer
        run_manager = RunManager()

        run_manager.begin_run(run, model, data_loader, device, tb_summary)

        torch.backends.cudnn.enabled = False

        # start training
        for epoch in range(epochs):
            run_manager.begin_epoch()

            for batch in data_loader:
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)

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
                run_manager.track_total_correct_per_epoch(predictions, labels)

            run_manager.end_epoch()
            loss_val.append(run_manager.get_final_loss_val())
            acc_val.append(run_manager.get_final_accuracy())

        run_manager.end_run()
        run_manager.save(save_file_name)
        if show_plot:
            self.plot_loss_val(loss_val, run)
            self.plot_accuracy_val(acc_val, run)

        return model

    @staticmethod
    def plot_loss_val(bn_loss, run):
        """
        Plots the graph based on the loss vs run parameter

        :param bn_loss: loss
        :param run:

        :return: none
        """
        plt.plot(bn_loss)
        plt.title(f'Fig: {run} Loss vs Epoch')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.show()

    @staticmethod
    def plot_accuracy_val(bn_acc, run):
        """
        Plots the accuracy graph based on the training accuracy score vs run parameter

        :param bn_acc:
        :param run:

        :return: none
        """
        plt.plot(bn_acc)
        plt.title(f'Fig: {run} Accuracy vs Epoch')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.show()

    def __getModel(self, type_of_model):
        """
        Initializes the cnn model based on the type

        :param type_of_model: whether {batch normalization, no batch normalization or dropout}

        :return: initialized model
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if type_of_model == "BatchNorm":
            return CNN_bn().to(device=device)
        elif type_of_model == "NoBatchNorm":
            return CNN_no_bn().to(device=device)
        elif type_of_model == "Dropout":
            return CNN_dropout().to(device=device)

    @staticmethod
    def __get_file_name(type_of_model, shuffle, lr, batch_size):
        """
        Gets the file name of the model if it exits in the disk

        :param type_of_model: whether {batch normalization, no batch normalization or dropout}
        :param shuffle:
        :param lr: learning rate
        :param batch_size: size of each batch

        :return: name of the file name of the model
        """
        if type_of_model == "BatchNorm":
            return "_with_bn_hwcr_cnn_lr_" + str(lr) + \
                   "_batch_size_" + str(batch_size) + "shuffle_" + str(shuffle)
        elif type_of_model == "NoBatchNorm":
            return "_without_bn_hwcr_cnn_lr_" + str(lr) + \
                   "_batch_size_" + str(batch_size) + "shuffle_" + str(shuffle)
        elif type_of_model == "Dropout":
            return "_with_bn_do_hwcr_cnn_lr_" + str(lr) + \
                   "_batch_size_" + str(batch_size) + "shuffle_" + str(shuffle)

    @staticmethod
    def __get_tb_summary_title(type_of_model):
        """
        Gets the summary file name

        :param type_of_model: whether {batch normalization, no batch normalization or dropout}

        :return: the summary file name
        """
        if type_of_model == "BatchNorm":
            return "With-Batch_Normalization-"
        elif type_of_model == "NoBatchNorm":
            return "Without-Batch_Normalization-"
        elif type_of_model == "Dropout":
            return "WithDropout_BN-"
