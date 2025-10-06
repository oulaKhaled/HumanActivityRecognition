import matplotlib.pyplot as plt


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
