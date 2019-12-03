from collections import OrderedDict

import torch

from HWCRUtils import HWCRUtils
from HandWrittenRecognitionDeep import HandWrittenRecognitionDeep


def test_with_diff_params():
    parameters = OrderedDict(
        lr=[0.001],
        batch_size=[70],
        shuffle=[False]
    )
    run = HWCRUtils.get_runs(parameters)

    data_set_path = "train_data.pkl"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_directory_path = './model/model'
    save_logistics_file_path = './metrics/'
    hwRD = HandWrittenRecognitionDeep()

    test_data_set = \
        hwRD.pre_process_test(data_set_path, device)

    model_path_bn = model_directory_path + "_no_bn_hwcr_cnn_lr" + str(run[0].lr) + \
                    "_batch_size_" + str(run[0].batch_size) + \
                    "shuffle_" + str(run[0].shuffle) + ".pt"

    response = hwRD.test_model_final(test_data_set, model_path_bn)
    if response:
        print(response)

    return response


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))

    print("Running on " + str(device))
    test_with_diff_params()
