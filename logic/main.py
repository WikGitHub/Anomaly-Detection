import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras import layers, Model

from logic.data_preprocessing import normal_train_data, test_data, normal_test_data, anomalous_test_data


class AnomalyDetector(Model):
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
                layers.Dense(140, activation="sigmoid")
            ]
        )

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


autoencoder = AnomalyDetector()
autoencoder.compile(optimizer="adam", loss="mae")
autoencoder.fit(normal_train_data, normal_train_data, epochs=20, validation_data=(test_data, test_data), shuffle=True)


# plt.plot(autoencoder.history.history["loss"], label="Training Loss")
# plt.plot(autoencoder.history.history["val_loss"], label="Validation Loss")
# plt.legend()
# plt.show()

#
encoded_imgs = autoencoder.encoder(normal_test_data).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

plt.plot(normal_test_data[0], label="Original")
plt.plot(decoded_imgs[0], label="Reconstructed")
plt.fill_between(np.arange(140), decoded_imgs[0], normal_test_data[0], color="lightcoral")
plt.legend(labels=["Original", "Reconstructed", "Error"])
plt.show()


# plot for anomalous data
# encoded_imgs = autoencoder.encoder(test_data).numpy()
# decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()
#
# plt.plot(test_data[0], label="Original")
# plt.plot(decoded_imgs[0], label="Reconstructed")
# plt.fill_between(np.arange(140), decoded_imgs[0], test_data[0], color="lightcoral")
# plt.legend(labels=["Original", "Reconstructed", "Error"])
# plt.show()


# show loss on histogram

# reconstructions = autoencoder.predict(normal_test_data)
# train_loss = tf.keras.losses.mae(reconstructions, normal_test_data)
#
# plt.hist(train_loss[None, :], bins=50)
# plt.xlabel("Train loss")
# plt.ylabel("No of examples")
# plt.show()

# threshold = np.mean(train_loss) + np.std(train_loss)
# print("Threshold: ", threshold)

#show loss of anomalous data on histogram
# reconstructions = autoencoder.predict(anomalous_test_data)
# train_loss = tf.keras.losses.mae(reconstructions, anomalous_test_data)
#
# plt.hist(train_loss[None, :], bins=50)
# plt.xlabel("Train loss")
# plt.ylabel("No of examples")
# plt.show()



# calculate accuracy and ect