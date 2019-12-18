import numpy as np
import torch

from HandWrittenRecognitionDeep import HandWrittenRecognitionDeep


def test_with_diff_params(device):
    """
    This method tests with real life data.(the hard dataset to detect all the 8 labels)
    :param device: {cpu and gpu}

    :return: the numpy array with all the predicted labels ’a’, ’b’,’c’,’d’,’h’, ’i’, ’j’ and ’k’
    """
    data_set_path = "HardData.pkl"
    hwRD = HandWrittenRecognitionDeep()

    test_data_set = hwRD.pre_process_test(data_set_path, device)
    model_path_bn = "./model/model_no_bn_hwcr_cnn_lr0.001_batch_size_64shuffle_False.pt"

    response = hwRD.test_model_final(test_data_set, model_path_bn)
    if response:
        np.save("hard_file.npy", response)
        print(response)

    return response


if __name__ == '__main__':
    _device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))

    print("Running on " + str(_device))
    test_with_diff_params(_device)
