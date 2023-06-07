import matplotlib.pyplot as plt
import numpy as np


def plot_ecg(data_to_plot, title) -> None:
    """
    Plots the ECG data
    :param data_to_plot: data to plot
    :param title: title of the plot
    """
    plt.plot(data_to_plot[0], c="red")
    plt.title(title)
    plt.show()


def plot_losses(data_to_plot) -> None:
    """
    Plots the losses
    :param data_to_plot: data to plot
    """
    plt.plot(data_to_plot.history["loss"])
    plt.plot(data_to_plot.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["loss", "val_loss"], loc="upper left")
    plt.show()
