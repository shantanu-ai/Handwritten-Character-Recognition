import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn import metrics

from HWCRUtils import HWCRUtils
from TestManager import Test_Manager
from TrainManager import Train_Manager


class HandWrittenRecognitionDeep:
    @staticmethod
    def split_train_test_validation_set(data_set_path, label_set_path, image_dims, split_size, device, flag):
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
        train_data_set1 = HWCRUtils.read_dataset_test(data_set_path)
        tensor_x = torch.stack([torch.Tensor(i) for i in train_data_set1])
        processed_dataset = torch.utils.data.TensorDataset(tensor_x)
        return processed_dataset

    @staticmethod
    def train_model(run, train_set, model_directory_path, model_paths, save_logistics_file_path, type, epochs=10,
                    show_plot=False):
        train = Train_Manager()
        return train.train_data_set(train_set, run, model_directory_path, model_paths, save_logistics_file_path,
                                    epochs, type, show_plot)

    def test_model(self, network, data_set, data_set_size, classes, run, type_of_bn,
                   device, show_confusion_matrix=False):
        test = Test_Manager()
        ret = test.test_data_set(data_set, network, run, classes)
        unknown_count = ret['unknown_count']
        # percent_correct = (ret['total_correct'] / (data_set_size - unknown_count)) * 100
        percent_correct = (ret['total_correct'] / data_set_size) * 100
        confusion_matrix = ret['confusion_matrix'][1:len(classes)+1, 1:len(classes)+1]
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
        test = Test_Manager()
        return test.test_model(test_data_set, model_path_bn)

    @staticmethod
    def __show_accuracy_class(confusion_matrix, classes):
        print('{0} - {1}'.format('Category', 'Accuracy'))
        for i, r in enumerate(confusion_matrix):
            print('{0} - {1}'.format(classes[i], r[i] / np.sum(r) * 100))

    @staticmethod
    def __plot_confusion_matrix(confusion_matrix, classes):
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.matshow(confusion_matrix, aspect='auto', vmin=0, vmax=5, cmap=plt.get_cmap('Blues'))
        plt.ylabel('Actual Category')
        plt.yticks(range(len(classes)), classes)
        plt.xlabel('Predicted Category')
        plt.xticks(range(len(classes)), classes)
        plt.show()

    @staticmethod
    def __print_confusion_matrix(confusion_matrix, classes):
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
        plt.plot(bn_accuracy)
        plt.ylabel("Accuracy")
        plt.xlabel("Run")
        plt.title(title)
        plt.show()

    @staticmethod
    def __plotROC_curve(actual, class_probabilities, which_class):
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
