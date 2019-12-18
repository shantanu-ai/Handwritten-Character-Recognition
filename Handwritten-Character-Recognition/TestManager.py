import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from HWCRUtils import HWCRUtils
from CNN_minusBN import Network as CNN_no_bn
torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True)


class Test_Manager:
    """
    This class tests the cnn model
    """
    def __init__(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = CNN_no_bn().to(device=device)

    def test_model(self, test_data_set, model_path):
        """
         This method tests the data set.

        :param test_data_set: test data set
        :param model_path: path of the model

        :return predicted_labels: the numpy array of the predicted labels
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if os.path.isfile(model_path):
            # load trained model parameters from disk
            print('Loaded model parameters from disk.')
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            return self.test_data_set_final(test_data_set)

        else:
            print('First run train.py to generate the model before testing!!')
            return

    def test_data_set_final(self, test_data_set):
        """
        Tests the data set based on real data set and returns
        the numpy array of the predicted labels.

        :param test_data_set: test data set

        :return predicted_labels: the numpy array of the predicted labels:
        """

        data_loader = torch.utils.data.DataLoader(
            test_data_set, num_workers=1, shuffle=False, pin_memory=True
        )
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        output = []
        idx = 1

        for batch in data_loader:
            images = batch[0]
            images = images.to(device)

            # forward propagation
            preds = self.model(images)
            _, predicted = torch.max(preds.data, 1)

            if predicted.data == 0:
                output.append(-1)
            else:
                output.append(predicted.item())

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            idx = idx + 1

        return output

    def test_data_set(self, test_set, network, run, classes):
        """
        Tests the model based on the validation set.

        :param test_set: validation data set
        :param network: cnn model whether {batch normalization, no batch normalization or dropout}
        :param run: run parameters
        :param classes: true labels

        :return: a python dictionary with all the metric of the model
        """
        confusion_matrix = np.zeros([len(classes)+1, len(classes)+1], int)
        # set batch size
        data_loader = torch.utils.data.DataLoader(
            test_set, num_workers=0, shuffle=False, pin_memory=True
        )

        # set optimizer - Adam
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # start training
        total_loss = 0
        total_correct = 0
        output = {}
        idx = 1
        unknown_count = 0

        for batch in data_loader:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            # forward propagation
            preds = network(images)
            _, predicted = torch.max(preds.data, 1)

            # estimate loss
            loss = F.cross_entropy(preds, labels)

            total_loss += loss.item()

            if predicted.data == 0:
                output[idx] = -1
                unknown_count = unknown_count + 1
            else:
                output[idx] = predicted.item()

            total_correct += HWCRUtils.get_num_correct(preds, labels)
            for i, l in enumerate(labels):
                confusion_matrix[l.item(), predicted[i].item()] += 1

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            idx = idx + 1

        return {
            "network": network,
            "total_loss": total_loss,
            "total_correct": total_correct,
            "confusion_matrix": confusion_matrix,
            "output": output,
            "unknown_count": unknown_count
        }

    def test_class_probabilities(self, model, device, test_set, batch_size, which_class):
        """
        Calculates the probability of a particular class required to plot the ROC Curve

        :param model: cnn model
        :param device: whether {cpu or gpu}
        :param test_set:
        :param batch_size:
        :param which_class:

        :return: the probability
        """
        model.eval()
        actuals = []
        data_loader = torch.utils.data.DataLoader(
            test_set, num_workers=1, shuffle=False, pin_memory=True
        )

        probabilities = []
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(device), target.to(device)
                if target.item() != 0:
                    output = model(data)
                    prediction = output.argmax(dim=1, keepdim=True)
                    actuals.extend(target.view_as(prediction) == which_class)
                    probabilities.extend(np.exp(output.data.cpu().numpy()[:, which_class]))

        return [i.item() for i in actuals], [i.item() for i in probabilities]
