from HWCRUtils import HWCRUtils
from Train import Train_Manager
from Test import Test_Manager
import matplotlib.pyplot as plt

import numpy as np


class HandWrittenRecognitionDeep:
    def __init__(self):
        self.__train_set = None
        self.__test_set = None
        self.__validation_set = None
        self.__test_set_size = None
        self.__validation_set_size = None

    def perform_hand_written_char_recognition(self, data_set_path, label_set_path, image_dims, parameters, split_size,
                                              model_directory_path, model_path, classes,
                                              epochs=10):
        self.__split_train_test_validation_set(data_set_path, label_set_path, image_dims, split_size)
        network = self.__train_model(parameters, model_directory_path, model_path,
                                     epochs)
        self.__test_model(network, classes, parameters)

    def __split_train_test_validation_set(self, data_set_path, label_set_path, image_dims, split_size):
        train_data_set, labels_set = HWCRUtils.read_dataset(data_set_path, label_set_path, image_dims)
        X_train, X_test, Y_train, Y_test = HWCRUtils.spilt_data_set(train_data_set,
                                                                    labels_set,
                                                                    split_size=split_size)
        X_train, X_val, Y_train, Y_val = HWCRUtils.spilt_data_set(X_train,
                                                                  Y_train,
                                                                  split_size=split_size)
        train_set = HWCRUtils.convert_to_tensor(X_train, Y_train)
        test_set = HWCRUtils.convert_to_tensor(X_test, Y_test)
        val_set = HWCRUtils.convert_to_tensor(X_val, Y_val)

        self.__train_set = train_set
        self.__test_set = test_set
        self.__validation_set = val_set
        self.__test_set_size = Y_test.shape[0]
        self.__validation_set_size = Y_test.shape[0]

    def __train_model(self, parameters, model_directory_path, model_path,
                      epochs=10):
        train = Train_Manager()
        response = train.train_data_set(self.__train_set,
                                        HWCRUtils.get_runs(parameters)[0],
                                        model_directory_path, model_path,
                                        epochs)
        return response["network"]

    def __test_model(self, network, classes, parameters):
        test = Test_Manager()
        ret = test.test_data_set(self.__validation_set, network, HWCRUtils.get_runs(parameters)[0])
        percent_correct = (ret['total_correct'] / self.__test_set_size) * 100
        confusion_matrix = ret['confusion_matrix'][1:9, 1:9]
        print(f"total loss test: {ret['total_loss']}")
        print(f"correctly predicted: {ret['total_correct']}")
        print(f"actual correct: {self.__test_set_size}")
        print(f"% correct: {percent_correct}")
        self.__show_accuracy_class(confusion_matrix, classes)
        self.__plot_confusion_matrix(confusion_matrix=confusion_matrix, classes=classes)
        self.__print_confusion_matrix(confusion_matrix, classes)

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
        plt.yticks(range(9), classes)
        plt.xlabel('Predicted Category')
        plt.xticks(range(8), classes)
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
