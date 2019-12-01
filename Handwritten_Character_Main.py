from collections import OrderedDict
import torch

from sklearn.model_selection import cross_val_score
from skorch import NeuralNetClassifier

from HWCRUtils import HWCRUtils
from HandWrittenRecognitionDeep import HandWrittenRecognitionDeep


def perform_CV(response, X_train, Y_train, cv, run, epochs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logistic_no_bn = NeuralNetClassifier(response["network_no_bn"], max_epochs=epochs, lr=run.lr, device=device)
    scores_no_bn = cross_val_score(logistic_no_bn, X_train, Y_train, cv=cv, scoring="accuracy")
    print("CV Score with No Batch Normalization")
    print(f'run: {run} | scores: {scores_no_bn.mean()}')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logistic_bn = NeuralNetClassifier(response["network_bn"], max_epochs=epochs, lr=run.lr, device=device)
    scores_bn = cross_val_score(logistic_bn, X_train, Y_train, cv=cv, scoring="accuracy")
    print("CV Score with Batch Normalization")
    print(f'run: {run} | scores: {scores_bn.mean()}')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logistic_dropout = NeuralNetClassifier(response["network_bn_dropout"], max_epochs=epochs, lr=run.lr, device=device)
    scores_dropout = cross_val_score(logistic_dropout, X_train, Y_train, cv=cv, scoring="accuracy")
    print("CV Score with No Batch Normalization")
    print(f'run: {run} | scores: {scores_dropout.mean()}')

    return scores_bn.mean(), scores_no_bn.mean(), scores_dropout.mean()


def test_with_diff_params():
    # parameters = OrderedDict(
    #     lr=[0.01, 0.001],
    #     batch_size=[64, 128],
    #     shuffle=[False]
    # )

    # parameters = OrderedDict(
    #     lr=[0.01],
    #     batch_size=[64, 128],
    #     shuffle=[False]
    # )

    parameters = OrderedDict(
        lr=[0.001],
        batch_size=[64],
        shuffle=[False]
    )
    run_list = HWCRUtils.get_runs(parameters)
    data_set_path = "train_data.pkl"
    label_set_path = "finalLabelsTrain.npy"
    # data_set_path = "./output_40000/data.npy"
    # label_set_path = "./output_40000/labels.npy"
    image_dims = (64, 64)
    epochs = 25
    # epochs = 1
    split_size = 0.03
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    cv = 10
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_directory_path = './model/model'
    save_logistics_file_path = './metrics/'
    hwRD = HandWrittenRecognitionDeep()
    X_train, Y_train, train_set, test_set, validation_set, validation_size, test_set_size, validation_set_size = \
        hwRD.split_train_test_validation_set(data_set_path, label_set_path, image_dims, split_size, device)

    for run in run_list:
        print("--------------------------------------------")
        print(run)
        model_path_no_bn = model_directory_path + "no_bn_hwcr_cnn_lr_" + str(run.lr) + \
                           "_batch_size_" + str(run.batch_size) + \
                           "shuffle_" + str(run.shuffle) + ".pt"

        model_path_bn = model_directory_path + "bn_hwcr_cnn_lr_" + str(run.lr) + \
                        "_batch_size_" + str(run.batch_size) + \
                        "shuffle_" + str(run.shuffle) + ".pt"

        model_path_dropout = model_directory_path + "bn_dropout_hwcr_cnn_lr_" + str(run.lr) + \
                             "_batch_size_" + str(run.batch_size) + \
                             "shuffle_" + str(run.shuffle) + ".pt"
        model_paths = (model_path_no_bn, model_path_bn, model_path_dropout)

        response = hwRD.train_model(run, train_set, model_directory_path, model_paths, save_logistics_file_path, epochs)

        #cv_scores_bn, cv_scores_no_bn, cv_scores_dropout = perform_CV(response, X_train, Y_train, cv, run, epochs)

        hwRD.test_model(response["network_bn"], validation_set, validation_size, classes, run,
                        "With Batch Normalization")
        hwRD.test_model(response["network_no_bn"], validation_set, validation_size, classes, run,
                        "Without Batch Normalization")
        hwRD.test_model(response["network_bn_dropout"], validation_set, validation_size, classes, run,
                        "With Dropout")

        print("-------------------------------------------")


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.get_device_name(0))
    test_with_diff_params()
