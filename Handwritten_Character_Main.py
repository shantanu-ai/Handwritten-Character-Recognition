from collections import OrderedDict

from sklearn.model_selection import cross_val_score
from skorch import NeuralNetClassifier

from HWCRUtils import HWCRUtils
from HandWrittenRecognitionDeep import HandWrittenRecognitionDeep


def perform_CV(response, X_train, Y_train, cv, run, epochs):
    logistic_bn = NeuralNetClassifier(response["network_bn"], max_epochs=epochs, lr=run.lr)
    scores_bn = cross_val_score(logistic_bn, X_train, Y_train, cv=cv, scoring="accuracy")
    print("CV Score with Batch Normalization")
    print(scores_bn.mean())

    logistic_no_bn = NeuralNetClassifier(response["network_no_bn"], max_epochs=epochs, lr=run.lr)
    scores_no_bn = cross_val_score(logistic_no_bn, X_train, Y_train, cv=cv, scoring="accuracy")
    print("CV Score with No Batch Normalization")
    print(scores_no_bn.mean())


def test_with_diff_params():
    # parameters = OrderedDict(
    #     lr=[0.01, 0.001, 0.03, 0.5],
    #     batch_size=[32, 64, 128, 256],
    #     shuffle=[False]
    # )

    # parameters = OrderedDict(
    #     lr=[0.01],
    #     batch_size=[64, 128],
    #     shuffle=[False]
    # )

    parameters = OrderedDict(
        lr=[0.01],
        batch_size=[64],
        shuffle=[False]
    )
    run_list = HWCRUtils.get_runs(parameters)
    data_set_path = "train_data.pkl"
    label_set_path = "finalLabelsTrain.npy"
    image_dims = (52, 52)
    epochs = 1
    # epochs = 1
    split_size = 0.03
    classes = [1, 2, 3, 4, 5, 6, 7, 8]
    cv = 10
    model_directory_path = '/Users/shantanughosh/Desktop/Shantanu_MS/Fall 19/FML/Project/Code_base/Handwritten-Character-Recognition/model/model'
    save_logistics_file_path = '/Users/shantanughosh/Desktop/Shantanu_MS/Fall 19/FML/Project/Code_base/Handwritten-Character-Recognition/metrics/'
    hwRD = HandWrittenRecognitionDeep()
    X_train, Y_train, train_set, test_set, validation_set, validation_size, test_set_size, validation_set_size = \
        hwRD.split_train_test_validation_set(data_set_path, label_set_path, image_dims, split_size)

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

        # perform_CV(response, X_train, Y_train, cv, run, epochs)

        hwRD.test_model(response["network_bn"], validation_set, validation_size, classes, run,
                        "With Batch Normalization")
        hwRD.test_model(response["network_no_bn"], validation_set, validation_size, classes, run,
                        "Without Batch Normalization")
        hwRD.test_model(response["network_bn_dropout"], validation_set, validation_size, classes, run,
                        "With Dropout")

        print("-------------------------------------------")


if __name__ == '__main__':
    test_with_diff_params()
