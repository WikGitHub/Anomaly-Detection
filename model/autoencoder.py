import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model


class AnomalyDetector(Model):
    """
    Anomaly detector model
    """

    def __init__(self):
        super(AnomalyDetector, self).__init__()
        self.encoder = tf.keras.Sequential(
            [
                layers.Dense(64, activation="relu"),
                layers.Dense(32, activation="relu"),
                layers.Dense(16, activation="relu"),
                layers.Dense(8, activation="relu"),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                layers.Dense(16, activation="relu"),
                layers.Dense(32, activation="relu"),
                layers.Dense(64, activation="relu"),
                layers.Dense(140, activation="sigmoid"),
            ]
        )

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        Calls the model, encoding and decoding the input data
        :param x: the input data
        :return: the decoded data
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded
