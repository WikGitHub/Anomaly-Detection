import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

dataframe = pd.read_csv("../data/ecg_data.csv")
raw_data = dataframe.values


labels = raw_data[:, -1]
data = raw_data[:, 0:-1]

# split data into train and test
from sklearn.model_selection import train_test_split

train_data, test_data, train_labels, test_labels = train_test_split(
    data, labels, test_size=0.2, random_state=21
)

# normalise data using min max tensor flow
min_val = tf.reduce_min(train_data)
max_val = tf.reduce_max(train_data)

train_data = (train_data - min_val) / (max_val - min_val)
test_data = (test_data - min_val) / (max_val - min_val)

train_data = tf.cast(train_data, tf.float32)
test_data = tf.cast(test_data, tf.float32)

train_labels = train_labels.astype(bool)
test_labels = test_labels.astype(bool)

normal_train_data = train_data[train_labels]
normal_test_data = test_data[test_labels]

anomalous_train_data = train_data[~train_labels]
anomalous_test_data = test_data[~test_labels]

# plt.grid()
# plt.plot(np.arange(140), normal_train_data[0])
# plt.title("A Normal ECG")
# plt.show()