from HWCRUtils import HWCRUtils
from Train import Train_Manager
from Test import Test_Manager
from CNN import Network


class HandWrittenRecognitionDeep:
    def __init__(self):
        self.__train_set = None
        self.__test_set = None
        self.__validation_set = None
        self.__test_set_size = None
        self.__validation_set_size = None

    def perform_hand_written_char_recognition(self, data_set_path, label_set_path, image_dims, parameters, split_size,
                                              epochs=10):
        self.__split_train_test_validation_set(data_set_path, label_set_path, image_dims, split_size)
        network = self.__train_model(parameters, epochs)
        self.__test_model(network, parameters)

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

    def __train_model(self, parameters, epochs=10):
        train = Train_Manager()
        network = Network()
        response = train.train_data_set(self.__train_set, network,
                                        HWCRUtils.get_runs(parameters)[0],
                                        epochs)
        return response["network"]

    def __test_model(self, network, parameters):
        test = Test_Manager()
        ret = test.test_data_set(self.__validation_set, network, HWCRUtils.get_runs(parameters)[0])
        percnt_correct = (ret['total_correct'] / self.__test_set_size) * 100
        print(f"total loss test: {ret['total_loss']}")
        print(f"correctly predicted: {ret['total_correct']}")
        print(f"actual correct: {self.__test_set_size}")
        print(f"% correct: {percnt_correct}")
