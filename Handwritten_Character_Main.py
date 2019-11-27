from collections import OrderedDict
from HandWrittenRecognitionDeep import HandWrittenRecognitionDeep
from HWCRUtils import HWCRUtils


def test_with_diff_params():
    # parameters = OrderedDict(
    #     lr=[0.01, 0.001, 0.03, 0.5],
    #     batch_size=[64, 128, 256],
    #     shuffle=[False]
    # )

    parameters = OrderedDict(
        lr=[0.01],
        batch_size=[64, 128],
        shuffle=[False]
    )
    run_list = HWCRUtils.get_runs(parameters)
    data_set_path = "train_data.pkl"
    label_set_path = "finalLabelsTrain.npy"
    image_dims = (52, 52)
    epochs = 10
    # epochs = 1
    split_size = 0.03
    classes = [1, 2, 3, 4, 5, 6, 7, 8]
    model_directory_path = '/Users/shantanughosh/Desktop/Shantanu_MS/Fall 19/FML/Project/Code_base/Handwritten-Character-Recognition/model/model'

    hwRD = HandWrittenRecognitionDeep()
    train_set, test_set, validation_set, validation_size, test_set_size, validation_set_size = \
        hwRD.split_train_test_validation_set(data_set_path, label_set_path, image_dims, split_size)

    for run in run_list:
        print(run)
        model_path = model_directory_path + "hwcr_cnn_lr" + str(run.lr) + \
                     "_batch_size_" + str(run.batch_size) + \
                     "shuffle_" + str(run.shuffle) + ".pt"
        network = hwRD.train_model(run, train_set, model_directory_path, model_path, epochs)
        hwRD.test_model(network, validation_set, validation_size, classes, run)


if __name__ == '__main__':
    test_with_diff_params()
    #
    # data_set_path = "train_data.pkl"
    # label_set_path = "finalLabelsTrain.npy"
    # image_dims = (52, 52)
    # epochs = 12
    # # epochs = 1
    # split_size = 0.10
    # parameters = OrderedDict(
    #     lr=[0.01],
    #     batch_size=[64]
    # )
    # classes = [1, 2, 3, 4, 5, 6, 7, 8]
    #
    # model_directory_path = '/Users/shantanughosh/Desktop/Shantanu_MS/Fall 19/FML/Project/Code_base/Handwritten-Character-Recognition/model'
    # model_path = model_directory_path + 'hwcr_cnn_lr_0.01_bs_64.pt'
    #
    # hwRD = HandWrittenRecognitionDeep()
    # hwRD.perform_hand_written_char_recognition(data_set_path, label_set_path, image_dims, parameters,
    #                                            split_size, model_directory_path, model_path, classes,
    #                                            epochs)
    #
    # ##################
    #
    # data_set_path = "train_data.pkl"
    # label_set_path = "finalLabelsTrain.npy"
    # image_dims = (52, 52)
    # epochs = 12
    # # epochs = 1
    # split_size = 0.10
    # parameters = OrderedDict(
    #     lr=[0.01],
    #     batch_size=[128]
    # )
    # classes = [1, 2, 3, 4, 5, 6, 7, 8]
    #
    # model_directory_path = '/Users/shantanughosh/Desktop/Shantanu_MS/Fall 19/FML/Project/Code_base/Handwritten-Character-Recognition/model'
    # model_path = model_directory_path + 'hwcr_cnn_lr_0.01_bs_128.pt'
    #
    # hwRD = HandWrittenRecognitionDeep()
    # hwRD.perform_hand_written_char_recognition(data_set_path, label_set_path, image_dims, parameters,
    #                                            split_size, model_directory_path, model_path, classes,
    #                                            epochs)
    #
    # ####################################################
    # data_set_path = "train_data.pkl"
    # label_set_path = "finalLabelsTrain.npy"
    # image_dims = (52, 52)
    # epochs = 12
    # # epochs = 1
    # split_size = 0.10
    # parameters = OrderedDict(
    #     lr=[0.01],
    #     batch_size=[32]
    # )
    # classes = [1, 2, 3, 4, 5, 6, 7, 8]
    #
    # model_directory_path = '/Users/shantanughosh/Desktop/Shantanu_MS/Fall 19/FML/Project/Code_base/Handwritten-Character-Recognition/model'
    # model_path = model_directory_path + 'hwcr_cnn_lr_0.01_bs_32.pt'
    #
    # hwRD = HandWrittenRecognitionDeep()
    # hwRD.perform_hand_written_char_recognition(data_set_path, label_set_path, image_dims, parameters,
    #                                            split_size, model_directory_path, model_path, classes,
    #                                            epochs)
    #
    # #############################################
    # data_set_path = "train_data.pkl"
    # label_set_path = "finalLabelsTrain.npy"
    # image_dims = (52, 52)
    # epochs = 12
    # # epochs = 1
    # split_size = 0.10
    # parameters = OrderedDict(
    #     lr=[0.001],
    #     batch_size=[64]
    # )
    # classes = [1, 2, 3, 4, 5, 6, 7, 8]
    #
    # model_directory_path = '/Users/shantanughosh/Desktop/Shantanu_MS/Fall 19/FML/Project/Code_base/Handwritten-Character-Recognition/model'
    # model_path = model_directory_path + 'hwcr_cnn_lr_0.001_bs_64.pt'
    #
    # hwRD = HandWrittenRecognitionDeep()
    # hwRD.perform_hand_written_char_recognition(data_set_path, label_set_path, image_dims, parameters,
    #                                            split_size, model_directory_path, model_path, classes,
    #                                            epochs)
    # ####################################
    # data_set_path = "train_data.pkl"
    # label_set_path = "finalLabelsTrain.npy"
    # image_dims = (52, 52)
    # epochs = 12
    # # epochs = 1
    # split_size = 0.10
    # parameters = OrderedDict(
    #     lr=[0.001],
    #     batch_size=[128]
    # )
    # classes = [1, 2, 3, 4, 5, 6, 7, 8]
    #
    # model_directory_path = '/Users/shantanughosh/Desktop/Shantanu_MS/Fall 19/FML/Project/Code_base/Handwritten-Character-Recognition/model'
    # model_path = model_directory_path + 'hwcr_cnn_lr_0.001_bs_128.pt'
    #
    # hwRD = HandWrittenRecognitionDeep()
    # hwRD.perform_hand_written_char_recognition(data_set_path, label_set_path, image_dims, parameters,
    #                                            split_size, model_directory_path, model_path, classes,
    #                                            epochs)
    #
    # ################################################
    # data_set_path = "train_data.pkl"
    # label_set_path = "finalLabelsTrain.npy"
    # image_dims = (52, 52)
    # epochs = 12
    # # epochs = 1
    # split_size = 0.10
    # parameters = OrderedDict(
    #     lr=[0.001],
    #     batch_size=[32]
    # )
    # classes = [1, 2, 3, 4, 5, 6, 7, 8]
    #
    # model_directory_path = '/Users/shantanughosh/Desktop/Shantanu_MS/Fall 19/FML/Project/Code_base/Handwritten-Character-Recognition/model'
    # model_path = model_directory_path + 'hwcr_cnn_lr_0.001_bs_32.pt'
    #
    # hwRD = HandWrittenRecognitionDeep()
    # hwRD.perform_hand_written_char_recognition(data_set_path, label_set_path, image_dims, parameters,
    #                                            split_size, model_directory_path, model_path, classes,
    #                                            epochs)
