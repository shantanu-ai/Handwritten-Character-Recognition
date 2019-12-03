from collections import OrderedDict

import torch

from HWCRUtils import HWCRUtils
from HandWrittenRecognitionDeep import HandWrittenRecognitionDeep


def final_test_train(train_set, test_set, run,
                     model_directory_path, save_logistics_file_path, hwRD, epochs,
                     test_set_size, classes):
    model_path_bn = model_directory_path + "_ab_no_bn_hwcr_cnn_lr" + str(run.lr) + \
                    "_batch_size_" + str(run.batch_size) + \
                    "shuffle_" + str(run.shuffle) + ".pt"

    model_bn = hwRD.train_model(run, train_set, model_directory_path, model_path_bn, save_logistics_file_path,
                                "NoBatchNorm", epochs, show_plot=True)
    response_test_bn = hwRD.test_model(model_bn, test_set, test_set_size, classes, run,
                                       "Without Batch Normalization", device, show_confusion_matrix=True)
    print(response_test_bn['accuracy'])


def test_with_diff_params():
    final_parameters = OrderedDict(
        lr=[0.001],
        batch_size=[64],
        shuffle=[False]
    )
    run_list = HWCRUtils.get_runs(final_parameters)
    data_set_path = "train_data.pkl"
    label_set_path = "finalLabelsTrain.npy"

    image_dims = (64, 64)
    epochs = 50
    # epochs = 1
    split_size = 0.03
    classes = ['a', 'b']
    ab_flag = True

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_directory_path = './model/model'
    save_logistics_file_path = './metrics/'
    hwRD = HandWrittenRecognitionDeep()
    X_train, Y_train, train_set, test_set, validation_set, validation_size, validation_set_size, test_set_size = \
        hwRD.split_train_test_validation_set(data_set_path, label_set_path, image_dims, split_size, device, ab_flag)

    bn_accuracy = []
    no_bn_accuracy = []
    dropout_accuracy = []

    # for run in run_list:
    #     print("--------------------------------------------")
    #     print(run)
    #     model_path_no_bn = model_directory_path + "no_bn_hwcr_cnn_lr_" + str(run.lr) + \
    #                        "_batch_size_" + str(run.batch_size) + \
    #                        "shuffle_" + str(run.shuffle) + ".pt"
    #
    #     model_path_bn = model_directory_path + "bn_hwcr_cnn_lr_" + str(run.lr) + \
    #                     "_batch_size_" + str(run.batch_size) + \
    #                     "shuffle_" + str(run.shuffle) + ".pt"
    #
    #     model_path_dropout = model_directory_path + "bn_dropout_hwcr_cnn_lr_" + str(run.lr) + \
    #                          "_batch_size_" + str(run.batch_size) + \
    #                          "shuffle_" + str(run.shuffle) + ".pt"
    #
    #     model_no_bn = hwRD.train_model(run, train_set, model_directory_path, model_path_no_bn, save_logistics_file_path,
    #                                    "NoBatchNorm", epochs)
    #
    #     model_bn = hwRD.train_model(run, train_set, model_directory_path, model_path_bn, save_logistics_file_path,
    #                                 "BatchNorm", epochs)
    #
    #     model_dropout = hwRD.train_model(run, train_set, model_directory_path, model_path_dropout,
    #                                      save_logistics_file_path,
    #                                      "Dropout", epochs)
    #
    #     # cv_scores_bn, cv_scores_no_bn, cv_scores_dropout = perform_CV(response, X_train, Y_train, cv, run, epochs)
    #
    #     response_test_bn = hwRD.test_model(model_bn, validation_set, validation_size, classes, run,
    #                                        "With Batch Normalization", device, show_confusion_matrix=False)
    #     response_test_no_bn = hwRD.test_model(model_no_bn, validation_set, validation_size, classes, run,
    #                                           "Without Batch Normalization", device, show_confusion_matrix=False)
    #     response_test_dropout = hwRD.test_model(model_dropout, validation_set, validation_size,
    #                                             classes,
    #                                             run,
    #                                             "With Dropout", device, show_confusion_matrix=False)
    #
    #     bn_accuracy.append(response_test_bn['accuracy'])
    #     no_bn_accuracy.append(response_test_no_bn['accuracy'])
    #     dropout_accuracy.append(response_test_dropout['accuracy'])
    #
    #     print("-------------------------------------------")

    # hwRD.plot_accuracy_run(sorted(bn_accuracy), "Batch Normalization")
    # hwRD.plot_accuracy_run(sorted(no_bn_accuracy), "Without Batch Normalization")
    # hwRD.plot_accuracy_run(sorted(dropout_accuracy), "With Dropout")



    final_test_train(train_set, test_set, run_list[0], model_directory_path, save_logistics_file_path,
                     hwRD, epochs, test_set_size, classes)


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))

    print("Running on " + str(device))
    test_with_diff_params()
