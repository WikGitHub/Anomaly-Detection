import matplotlib.pyplot as plt
import tensorflow as tf

from logic.anomaly_detector import normal_train_data, anomalous_train_data


def plot_histogram(data: tf.Tensor):
    """
    Plot a histogram of the data
    :param data:  data to plot
    """
    plt.hist(data, bins=10)
    plt.xlabel("Values")
    plt.ylabel("Frequency")
    plt.title("Histogram of Data Distribution")
    plt.show()


def plot_scatter(normal_data: tf.Tensor, anomalous_data: tf.Tensor):
    """
    Plot a scatter plot of the data
    :param normal_data: data of a normal ecg
    :param anomalous_data: data of an anomalous ecg
    """
    plt.scatter(normal_data[:, 0], normal_data[:, 1], color="blue", label="Normal Data")
    plt.scatter(
        anomalous_data[:, 0], anomalous_data[:, 1], color="red", label="Anomalous Data"
    )
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Scatter Plot of Normal and Anomalous Data")
    plt.legend()
    plt.show()


plot_histogram(normal_train_data)
plot_histogram(anomalous_train_data)
plot_scatter(normal_train_data, anomalous_train_data)


# TODO: plot training vs validation_loss
# TODO: better visualisations of the data
