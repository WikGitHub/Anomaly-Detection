import tf
from tensorflow.keras import layers


class AnomalyDetector(tf.keras.Model):
    """
    Anomaly detector model
    """

    def __init__(self):
        super(AnomalyDetector, self).__init__()
        self.encoder = tf.keras.Sequential(
            [
                layers.Dense(32, activation="relu"),
                layers.Dense(16, activation="relu"),
                layers.Dense(8, activation="relu"),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                layers.Dense(16, activation="relu"),
                layers.Dense(32, activation="relu"),
                layers.Dense(140, activation="sigmoid"),
            ]
        )

    def call(self, input_data: tf.Tensor) -> tf.Tensor:
        """
        Calls the model, encoding and decoding the input data
        :param input_data: the input data
        :return: the decoded data
        """
        encoded = self.encoder(input_data)
        decoded = self.decoder(encoded)
        return decoded
