import os
import csv
from datetime import datetime

class ExperimentLoggerCSV:
    def __init__(self, log_dir="./logs", log_file="experiment_log.csv"):

        self.log_dir = log_dir
        self.log_file = log_file
        self._ensure_log_dir_exists()
        self._initialize_log_file()

    def _ensure_log_dir_exists(self):

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def _initialize_log_file(self):

        log_path = os.path.join(self.log_dir, self.log_file)
        if not os.path.exists(log_path):
            with open(log_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "experiment_name", "round",  "test_accuracy", "dataset", "model", "learning_rate","local_learning_rate"
                ])

    def log(self, experiment_name, metrics, additional_info=None):

        log_path = os.path.join(self.log_dir, self.log_file)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


        log_data = {
            "timestamp": timestamp,
            "experiment_name": experiment_name,
            **metrics,
            **(additional_info if additional_info else {})
        }


        with open(log_path, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "timestamp", "experiment_name", "round",  "test_accuracy", "dataset", "model", "learning_rate","local_learning_rate"
            ])
            writer.writerow(log_data)

        print(f"log adding at {log_path}")