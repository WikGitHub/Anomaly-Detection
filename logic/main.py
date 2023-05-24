import tensorflow as tf
from tensorflow.keras import layers, Model

(x_train, _), _ = tf.keras.datasets.mnist.load_data()
x_train_preprocessed = x_train.reshape((-1, 784)) / 255.0


class DigitEncoder(Model):
    def __init__(self):
        super(DigitEncoder, self).__init__()
        self.encoder = tf.keras.Sequential(
            [
                layers.Dense(128, activation="relu"),
                layers.Dense(64, activation="relu"),
                layers.Dense(32, activation="relu"),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                layers.Dense(64, activation="relu"),
                layers.Dense(128, activation="relu"),
                layers.Dense(784, activation="relu"),
            ]
        )

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


autoencoder = DigitEncoder()

autoencoder.compile(optimizer="adam", loss="mean_squared_error")

autoencoder.fit(x_train_preprocessed, x_train_preprocessed, epochs=1, batch_size=32)

encoded_data = autoencoder.encoder(x_train_preprocessed)
decoded_data = autoencoder.decoder(encoded_data)
