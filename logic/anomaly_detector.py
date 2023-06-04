import tensorflow as tf

from _logging.main import get_logger
from logic.data_preprocessing import preprocess_data
from model.autoencoder import AnomalyDetector

_logger = get_logger(__name__)


def train_autoencoder(
    autoencoder: AnomalyDetector,
    train_data: tf.Tensor,
    epochs: int = 20,
    validation_data: tf.Tensor = None,
) -> None:
    """
    Train the autoencoder model on the train data
    :param autoencoder: the autoencoder model
    :param train_data: the train data
    :param epochs: the number of epochs to train for
    :param validation_data: the validation data
    """
    autoencoder.compile(optimizer="adam", loss="mae")
    autoencoder.fit(
        train_data,
        train_data,
        epochs=epochs,
        validation_data=validation_data,
        shuffle=True,
    )


def evaluate_autoencoder(autoencoder: AnomalyDetector, test_data: tf.Tensor) -> None:
    """
    Evaluate the autoencoder model on the test data
    :param autoencoder: the autoencoder model
    :param test_data: the test data
    """
    loss = autoencoder.evaluate(test_data, test_data)
    _logger.info("The loss is: {}".format(loss))


def productionise_autoencoder(
    autoencoder: AnomalyDetector,
    train_data: tf.Tensor,
    test_data: tf.Tensor,
    epochs: int = 20,
) -> None:
    """
    Productionise the autoencoder model
    :param autoencoder: the autoencoder model
    :param train_data: the train data
    :param test_data: the test data
    :param epochs: the number of epochs to train for
    """
    train_autoencoder(autoencoder, train_data, epochs)
    evaluate_autoencoder(autoencoder, test_data)


# Preprocess the data
(
    normal_train_data,
    anomalous_train_data,
    normal_test_data,
    anomalous_test_data,
) = preprocess_data("../data/ecg_data.csv")

# Create an instance of the AnomalyDetector model
autoencoder = AnomalyDetector()

# Train and evaluate the autoencoder
productionise_autoencoder(autoencoder, normal_train_data, normal_test_data, epochs=20)
