from collections import OrderedDict
from HandWrittenRecognitionDeep import HandWrittenRecognitionDeep

if __name__ == '__main__':
    data_set_path = "train_data.pkl"
    label_set_path = "finalLabelsTrain.npy"
    image_dims = (52, 52)
    epochs = 12
    # epochs = 1
    split_size = 0.10
    parameters = OrderedDict(
        lr=[0.01],
        batch_size=[64]
    )
    classes = [1, 2, 3, 4, 5, 6, 7, 8]

    model_directory_path = '/Users/shantanughosh/Desktop/Shantanu_MS/Fall 19/FML/Project/Code_base/Handwritten-Character-Recognition/model'
    model_path = model_directory_path + 'hwcr_cnn.pt'

    hwRD = HandWrittenRecognitionDeep()
    hwRD.perform_hand_written_char_recognition(data_set_path, label_set_path, image_dims, parameters,
                                               split_size, model_directory_path, model_path, classes,
                                               epochs)
