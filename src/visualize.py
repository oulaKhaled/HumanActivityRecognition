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
    plt.ylabel("Sensor Signal for one sample")s
