import time
import torchvision
import json
import pandas as pd

from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter

from HWCRUtils import HWCRUtils


class RunManager:
    def __init__(self):
        self.epoch_id = 0
        self.epoch_loss = 0
        self.epoch_id_total_correct = 0
        self.epoch_id_actual_correct = 0
        self.epoch_start_time = None

        self.run_params = None
        self.run_id = 0
        self.run_data = []
        self.run_start_time = None

        self.network = None
        self.loader = None
        self.tb = None

    def begin_run(self, run, network, loader, type_of_bn):
        self.run_start_time = time.time()

        self.run_id += 1
        self.run_params = run
        self.network = network
        self.loader = loader
        self.tb = SummaryWriter(comment=f'{type_of_bn} -{run}')
        images, labels = next(iter(self.loader))
        grid = torchvision.utils.make_grid(images)
        self.tb.add_image("images", grid)
        self.tb.add_graph(network, images)

    def end_run(self):
        self.tb.close()
        self.epoch_id = 0

    def begin_epoch(self):
        self.epoch_start_time = time.time()

        self.epoch_id += 1
        self.epoch_loss = 0
        self.epoch_id_actual_correct = 0
        self.epoch_id_total_correct = 0

    def end_epoch(self):
        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time

        loss = self.epoch_loss / len(self.loader.dataset)
        accuracy = self.epoch_id_total_correct / len(self.loader.dataset)
        self.tb.add_scalar("Loss", loss, self.epoch_id)
        self.tb.add_scalar("Number Correct", self.epoch_id_total_correct, self.epoch_id)
        self.tb.add_scalar("Accuracy", accuracy, self.epoch_id)
        for name, param in self.network.named_parameters():
            self.tb.add_histogram(name, param, self.epoch_id)
            self.tb.add_histogram(f'{name}.grad', param.grad, self.epoch_id)

        results = OrderedDict()
        results["run"] = self.run_id
        results["epoch"] = self.epoch_id
        results["loss"] = loss
        results["accuracy"] = accuracy
        results["epoch_duration"] = epoch_duration
        results["run_duration"] = run_duration

        for k, v in self.run_params._asdict().items():
            results[k] = v
        self.run_data.append(results)
        df = pd.DataFrame.from_dict(self.run_data, orient='columns')
        print(df)

    def track_loss(self, loss):
        self.epoch_loss += loss.item() * self.loader.batch_size

    def track_total_correct_per_epoch(self, preds, labels):
        self.epoch_id_total_correct += HWCRUtils.get_num_correct(preds, labels)

    def save(self, fileName):
        pd.DataFrame.from_dict(
            self.run_data,
            orient='columns'
        ).to_csv(f'{fileName}.csv')

        with open(f'{fileName}.json', 'w', encoding="utf-8") as f:
            json.dump(self.run_data, f, ensure_ascii=False, indent=4)
0