from collections import OrderedDict
from HandWrittenRecognitionDeep import HandWrittenRecognitionDeep

if __name__ == '__main__':
    data_set_path = "train_data.pkl"
    label_set_path = "finalLabelsTrain.npy"
    image_dims = (52, 52)
    epochs = 10
    split_size = 0.01
    parameters = OrderedDict(
        lr=[0.01],
        batch_size=[100]
    )

    hwRD = HandWrittenRecognitionDeep()
    hwRD.perform_hand_written_char_recognition(data_set_path, label_set_path, image_dims, parameters, split_size,
                                               epochs)
