import matplotlib.pyplot as plt
import pandas as pd


def plot_single_data_point_info(_train_data, sample_index, signals):
    """
    understand the shape, range, and periodicity of the raw signals.
    """
    sample = _train_data[sample_index]
    plt.figure(figsize=(10, 6))
    for signal, i in zip(signals, range(len(signals))):
        plt.plot(sample[:, i], label=signal)
    plt.title("Single Sample Visualization")
    plt.xlabel("Time steps ")
    plt.ylabel("Sensor Signal for one sample")


def plot_model_results(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].plot(history.history["accuracy"], label="Training accuracy")
    axes[0].plot(history.history["val_accuracy"], label="validation accuracy")
    axes[0].set_title("Training accuracy and Validation accuracy")
    axes[0].set_ylabel("accuracy")
    axes[0].set_xlabel("epochs")
    axes[1].plot(history.history["loss"], label="Training Loss")
    axes[1].plot(history.history["val_loss"], label="Validation Loss")
    axes[1].set_title("Training and Validation Loss")
    axes[1].set_ylabel("Loss")
    axes[1].set_xlabel("epochs")
    plt.tight_layout()
    plt.legend()
    plt.show()


def get_sample(sample_from_each_group, activity_index, sensor_index, x_train):
    """
    pick a channel for specific activity
    """
    ## get sample index
    _activity_sample_index = sample_from_each_group.index[activity_index]
    _sensor_val = x_train[_activity_sample_index][
        :, sensor_index
    ]  # get channel value coresponding to sample index
    return _sensor_val  # 128 value for each time step for a specific channel to specific activity


def get_all_activites(samples_from_each_group, activity_indices, sensor_index, x_train):
    """
    get channel value for all activites for one sample
    """
    return [
        get_sample(samples_from_each_group, i, x_train, sensor_index)
        for i in activity_indices
    ]


def plot_signal_signal_across_activity(y_train, x_train, activity_labels):
    y_train = pd.DataFrame(y_train)
    ## get one sample from each group
    sample_each_group = y_train.groupby(0).sample(n=1)
    sample_list = get_all_activites(
        samples_from_each_group=sample_each_group,
        x_train=x_train,
        activity_indices=[0, 1, 2, 3, 4, 5],
        sensor_index=0,
    )
    for i, sample in enumerate(sample_list):
        plt.plot(sample, label=f"Body Acceleration for {activity_labels[i]} activity ")
        plt.ylabel("body_acc_x values ")
        plt.xlabel("Time Steps across each body_acc_x")
        plt.legend(loc="upper right", title_fontsize=5)
        plt.title("Different activity comparison")
