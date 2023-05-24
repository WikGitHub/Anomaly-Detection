import tensorflow as tf
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt
import numpy as np

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


# Create an instance of the DigitEncoder model
autoencoder = DigitEncoder()

# Compile the model
autoencoder.compile(optimizer="adam", loss="mean_squared_error")

# Train the model
autoencoder.fit(x_train_preprocessed, x_train_preprocessed, epochs=10, batch_size=32)

# Encode and decode the data
encoded_data = autoencoder.encoder(x_train_preprocessed)
decoded_data = autoencoder.decoder(encoded_data)

# Convert EagerTensor to NumPy array and reshape
decoded_images = decoded_data.numpy().reshape((-1, 28, 28))

# Select a few random images for visualization
num_images = 5
random_indices = np.random.choice(len(x_train), num_images, replace=False)

# Plot the original and reconstructed images
fig, axes = plt.subplots(nrows=2, ncols=num_images, figsize=(10, 4))
for i, idx in enumerate(random_indices):
    axes[0, i].imshow(x_train[idx], cmap="gray")
    axes[0, i].axis("off")
    axes[0, i].set_title("Original")

    axes[1, i].imshow(decoded_images[idx], cmap="gray")
    axes[1, i].axis("off")
    axes[1, i].set_title("Reconstructed")

plt.tight_layout()
plt.show()
