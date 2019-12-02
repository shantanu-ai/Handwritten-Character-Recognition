import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from HWCRUtils import HWCRUtils

torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True)


class Test_Manager:
    def test_data_set(self, test_set, network, run):
        confusion_matrix = np.zeros([9, 9], int)
        batch_size = run.batch_size
        lr = run.lr

        # set batch size
        data_loader = torch.utils.data.DataLoader(
            test_set, num_workers=1, shuffle=False, pin_memory=True
        )

        # set optimizer - Adam
        optimizer = optim.Adam(network.parameters(), lr=lr)
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
                unknown_count = unknown_count + 0
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
