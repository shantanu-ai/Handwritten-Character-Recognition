import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from skorch import NeuralNetClassifier

from HWCRUtils import HWCRUtils
from TestManager import Test_Manager
from TrainManager import Train_Manager


class HandWrittenRecognitionDeep:
    """
    This class acts as a container which invokes methods that ->
        1. train models
        2. test models
        3. split the dataset to train, test and validation set
        4. perform cross validation
        5. plot confusion matrix
        6. plot the curves based on the different parameters
    """

    def train_model(self, run, cv_set, train_set, model_directory_path, model_paths, save_logistics_file_path,
                    type, cv, epochs=10, show_plot=False):
        """
        This method trains the model based on the parameters and returns the trained model.

        :param run: run parameters
        :param cv_set: cross validation data set
        :param train_set: training set
        :param model_directory_path: path in the disk where pre-trained model exists if any
        :param model_paths: path in the disk where trained model will be saved.
        :param save_logistics_file_path: logistics path where details about model will be saved
        :param type: type of model
        :param cv: cross validation size
        :param epochs:
        :param show_plot:

        :return model: pytorch model

        """
        train = Train_Manager()

        model = train.train_data_set(train_set, run, model_directory_path, model_paths, save_logistics_file_path,
                                     epochs, type, show_plot)
        X_train = cv_set[0]
        Y_train = cv_set[1]
        self.__perform_CV(model, X_train, Y_train, cv,
                          run, epochs, type)

        return model

    @staticmethod
    def split_train_test_validation_set(data_set_path, label_set_path, image_dims, split_size, device, flag):
        """
        This method splits the data set into train, test and validation set. Also this method resize the images
        based on image dimensions specified by image_dims parameter.

        :param data_set_path:
        :param label_set_path:
        :param image_dims:
        :param split_size:
        :param device:
        :param flag:

        :return train, test and validation set and their corresponding sizes
        """
        # train_data_set, labels_set = HWCRUtils.read_dataset(data_set_path, label_set_path, image_dims)
        custom_data_set, custom_labels_set = HWCRUtils.read_dataset("./output_40000/data.npy",
                                                                    "./output_40000/labels.npy", image_dims)
        train_data_set1, labels_set1 = HWCRUtils.read_dataset(data_set_path, label_set_path, image_dims)
        train_data_set = np.concatenate((train_data_set1, custom_data_set), axis=0)
        labels_set = np.concatenate((labels_set1, custom_labels_set), axis=0)
        if flag:
            ab_image_set = []
            ab_labels_set = []
            for i in range(labels_set.shape[0]):
                if labels_set[i] == 1 or labels_set[i] == 2:
                    ab_image_set.append(train_data_set[i])
                    ab_labels_set.append(labels_set[i])
            train_data_set = np.array(ab_image_set)
            labels_set = np.array(ab_labels_set)
        X_train, X_test, Y_train, Y_test = HWCRUtils.spilt_data_set(train_data_set,
                                                                    labels_set,
                                                                    split_size=split_size)
        X_train, X_val, Y_train, Y_val = HWCRUtils.spilt_data_set(X_train,
                                                                  Y_train,
                                                                  split_size=split_size)
        train_set = HWCRUtils.convert_to_tensor(X_train, Y_train, device)
        test_set = HWCRUtils.convert_to_tensor(X_test, Y_test, device)
        val_set = HWCRUtils.convert_to_tensor(X_val, Y_val, device)

        return X_train, Y_train, train_set, test_set, val_set, Y_val.shape[0], Y_val.shape[0], Y_test.shape[0]

    @staticmethod
    def pre_process_test(data_set_path, device):
        """
        This method converts the numpy array to tensors.

        :param data_set_path:
        :param device:

        :return: tensor
        """
        train_data_set1 = HWCRUtils.read_dataset_test(data_set_path)
        tensor_x = torch.stack([torch.Tensor(i) for i in train_data_set1])
        processed_dataset = torch.utils.data.TensorDataset(tensor_x)
        return processed_dataset

    def test_model(self, network, data_set, data_set_size, classes, run, type_of_bn,
                   device, show_confusion_matrix=False):
        """
        This method tests the model based on validation set and prints the accuracy score
        of the model. It also prints the confusion matrix if the user wants to.

        :param network: cnn model
        :param data_set: validation data set
        :param data_set_size: test data set size
        :param classes: labels of the dataset
        :param run: run parameters
        :param type_of_bn: type of the cnn model
        :param device: device - either cpu or gpu
        :param show_confusion_matrix:

        :return dictionary: a dictionary with loss and accuracy of the model
        """
        test = Test_Manager()
        ret = test.test_data_set(data_set, network, run, classes)
        unknown_count = ret['unknown_count']
        # percent_correct = (ret['total_correct'] / (data_set_size - unknown_count)) * 100
        percent_correct = (ret['total_correct'] / data_set_size) * 100
        confusion_matrix = ret['confusion_matrix'][1:len(classes) + 1, 1:len(classes) + 1]
        print(f"#### {type_of_bn} #####")
        print(f"total loss test: {ret['total_loss']}")
        print(f"unknown_count: {unknown_count}")
        print(f"correctly predicted: {ret['total_correct']}")
        print(f"actual correct: {data_set_size}")
        print(f"% correct: {percent_correct}")
        self.__show_accuracy_class(confusion_matrix, classes)

        if show_confusion_matrix:
            self.__plot_confusion_matrix(confusion_matrix=confusion_matrix, classes=classes)
            self.__print_confusion_matrix(confusion_matrix, classes)
            # actual, class_probabilities = test.test_class_probabilities(network, device, data_set,
            #                                                             run.batch_size, which_class=len(classes))
            # self.__plotROC_curve(actual, class_probabilities, which_class=len(classes))

        return {
            "loss": ret['total_loss'],
            "accuracy": percent_correct
        }

    @staticmethod
    def test_model_final(test_data_set, model_path_bn):
        """
        Test the model based on the real test dataset
        :param test_data_set: test data set
        :param model_path_bn: path of the model in the disk

        :return predicted_labels: numpy array with the predicted labels
        """
        test = Test_Manager()
        return test.test_model(test_data_set, model_path_bn)

    @staticmethod
    def __show_accuracy_class(confusion_matrix, classes):
        """
        Shows the accuracy of the each true labels based on the confusion matrix.
        :param confusion_matrix:
        :param classes: true labels

        :return: none
        """
        print('{0} - {1}'.format('Category', 'Accuracy'))
        for i, r in enumerate(confusion_matrix):
            print('{0} - {1}'.format(classes[i], r[i] / np.sum(r) * 100))

    @staticmethod
    def __plot_confusion_matrix(confusion_matrix, classes):
        """
        Plots the confusion matrix.

        :param confusion_matrix:
        :param classes: true labels

        :return: none
        """
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.matshow(confusion_matrix, aspect='auto', vmin=0, vmax=5, cmap=plt.get_cmap('Blues'))
        plt.ylabel('Actual Category')
        plt.yticks(range(len(classes)), classes)
        plt.xlabel('Predicted Category')
        plt.xticks(range(len(classes)), classes)
        plt.show()

    @staticmethod
    def __print_confusion_matrix(confusion_matrix, classes):
        """
        Prints the confusion matrix in the console

        :param confusion_matrix:
        :param classes: true labels

        :return: none
        """
        print('actual/pred'.ljust(16), end='')
        for i, c in enumerate(classes):
            print(str(c).ljust(10), end='')
        print()
        for i, r in enumerate(confusion_matrix):
            print(str(classes[i]).ljust(16), end='')
            for idx, p in enumerate(r):
                print(str(p).ljust(10), end='')
            print()

            r = r / np.sum(r)
            print(''.ljust(16), end='')
            for idx, p in enumerate(r):
                print(str(p).ljust(10), end='')
            print()

    @staticmethod
    def plot_accuracy_run(bn_accuracy, title):
        """
        Plots the curve based on accuracy and run data

        :param bn_accuracy: accuracy of the model
        :param title: title of the plot

        :return: none
        """
        plt.plot(bn_accuracy)
        plt.ylabel("Accuracy")
        plt.xlabel("Run")
        plt.title(title)
        plt.show()

    @staticmethod
    def __plotROC_curve(actual, class_probabilities, which_class):
        """
        Plots the ROC curve of a particular label

        :param actual: true label
        :param class_probabilities:
        :param which_class: class

        :return: none
        """
        fpr, tpr, _ = metrics.roc_curve(actual, class_probabilities)
        roc_auc = metrics.auc(fpr, tpr)
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC for character=%d class' % which_class)
        plt.legend(loc="lower right")
        plt.show()

    @staticmethod
    def __perform_CV(model, X_train, Y_train, cv, run, epochs, type_of_model):
        """
        Performs the k fold cross validation based on the training data using skorch library.

        :param model: cnn model
        :param X_train: training data set
        :param Y_train: true class labels
        :param cv: no of folds
        :param run: run parameter
        :param epochs:
        :param type_of_model: whether {batch normalization, no batch normalization or dropout}

        :return: cross validation score
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logistics = NeuralNetClassifier(model, max_epochs=epochs, lr=run.lr, device=device)
        scores = cross_val_score(logistics, X_train, Y_train, cv=cv, scoring="accuracy")
        print(f'CV score with type {type_of_model}')
        print(f'run: {run} | score: {scores.mean()}')

        return scores.mean()
