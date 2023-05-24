# Convert EagerTensor to NumPy array and reshape
import numpy as np
from matplotlib import pyplot as plt

from logic.main import decoded_data, x_train

decoded_images = decoded_data.numpy().reshape((-1, 28, 28))

num_images = 5
random_indices = np.random.choice(len(x_train), num_images, replace=False)

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
