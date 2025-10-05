import pandas as pd
import numpy as np
import tensorflow as tf
import keras
import os
import re


MAIN_PATH = os.getenv("MAIN_PATH")
DATA_FOLDER_PATH = f"{MAIN_PATH}/UCI HAR Dataset/UCI HAR Dataset"
os.listdir(DATA_FOLDER_PATH)
TRAIN_FOLDER_PATH = f"{DATA_FOLDER_PATH}/train"
TEST_FOLDER_PATH = f"{DATA_FOLDER_PATH}/test"
os.listdir(TRAIN_FOLDER_PATH)


def read_file(file_path):
    with open(file_path, "r") as f:
        new_var = f.read().strip().split("\n")
    return new_var


def data_info(data_folder_path):
    columns_names = read_file(f"{data_folder_path}/features.txt")
    activity_labels = read_file(f"{data_folder_path}/activity_labels.txt")
    columns_names = [re.sub(r"\d+", "", i).strip() for i in columns_names]
    activity_labels2 = [re.sub(r"\d+", "", i).strip() for i in activity_labels]

    print(f"Sample from Features names {columns_names[0:5]}")
    print(f"Features Length : {len(columns_names)}")
    print(f"Target labels {activity_labels}")

    return (
        columns_names,
        activity_labels,
        activity_labels2,
    )


SIGNALS = [
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
]


def prepare_signal_data(signals, FOLDER_NAME, subset):
    data = []
    for i in signals:
        # print(f"{i} \n")
        df = pd.read_csv(
            f"{FOLDER_NAME}/Inertial Signals/{i}_{subset}.txt", sep=r"\s+", header=None
        )
        data.append(df.to_numpy())
    return np.transpose(data, (1, 2, 0))


def encode_labels(path):
    labels = pd.read_csv(path, sep=r"\s+", header=None)[0]
    return pd.get_dummies(labels).to_numpy()


def handle_signalData(signals, test_folder_path, train_folder_path):
    # Signals Data
    _test_data = prepare_signal_data(signals, test_folder_path, "test")
    _train_data = prepare_signal_data(signals, train_folder_path, "train")

    # Labels
    _train_labels = encode_labels(f"{train_folder_path}/y_train.txt")
    _test_labels = encode_labels(f"{test_folder_path}/y_test.txt")

    print(
        f"train data shape {_train_data.shape}\n tests data shape {_test_data.shape}\n"
    )
    print(f" train labels  {len(_train_labels)} test labels  {len(_test_labels)}")
    return _train_data, _train_labels, _test_data, _test_labels
