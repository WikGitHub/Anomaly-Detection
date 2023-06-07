import tensorflow as tf

from _logging.main import get_logger
from logic.data_preprocessing import preprocess_data
from logic.output_visualisation import plot_ecg, plot_losses
from model.autoencoder import AnomalyDetector

_logger = get_logger(__name__)


def train_autoencoder(
    autoencoder: AnomalyDetector,
    train_data: tf.Tensor,
    normal_test_data: tf.Tensor,
    epochs: int,
) -> tf.keras.callbacks.History:
    """
    Train the autoencoder model on the train data
    :param normal_test_data: the normal test data
    :param autoencoder: the autoencoder model
    :param train_data: the train data
    :param epochs: the number of epochs to train for
    :return: the history of the training
    """
    autoencoder.compile(optimizer="adam", loss="mse")
    history = autoencoder.fit(
        train_data,
        train_data,
        epochs=epochs,
        batch_size=256,
        validation_data=(normal_test_data, normal_test_data),
    )

    return history


def main(
    autoencoder: AnomalyDetector,
    epochs: int,
) -> None:
    """
    Train and evaluate the autoencoder model
    :param autoencoder: the autoencoder model
    :param epochs: the number of epochs to train for
    """
    # Preprocess the data
    (
        normal_train_data,
        anomalous_train_data,
        normal_test_data,
        anomalous_test_data,
    ) = preprocess_data("../data/ecg_data.csv")

    # Train and evaluate the autoencoder
    train_losses = train_autoencoder(
        autoencoder, normal_train_data, normal_test_data, epochs=epochs
    )

    plot_ecg(normal_train_data, title="Normal ECG")
    plot_ecg(anomalous_train_data, title="Anomalous ECG")
    plot_losses(train_losses)


if __name__ == "__main__":
    # Create the autoencoder model
    autoencoder = AnomalyDetector()

    # Train and evaluate the model
    main(autoencoder, epochs=50)
