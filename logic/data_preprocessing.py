from typing import Tuple

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


def load_data(file_path: str) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Load data from csv file and return data and labels as tensors
    :param file_path: path to csv file
    :return: data and labels as tensors
    """
    dataframe = pd.read_csv(file_path)
    data = dataframe.iloc[:, :-1].values
    labels = dataframe.iloc[:, -1].values.astype(bool)

    return data, labels


def normalise_data(
    test_data: tf.Tensor, train_data: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Normalise data to values between 0 and 1 using min-max normalisation
    :param test_data: test data to normalise
    :param train_data: train data to normalise
    :return: normalised data
    """
    min_value = tf.reduce_min(train_data)
    max_value = tf.reduce_max(train_data)

    train_data = (train_data - min_value) / (max_value - min_value)
    test_data = (test_data - min_value) / (max_value - min_value)

    train_data = tf.cast(train_data, dtype=tf.float32)
    test_data = tf.cast(test_data, dtype=tf.float32)

    return train_data, test_data


def separate_data_by_labels(
    data: tf.Tensor, labels: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Separate data by labels into normal and anomalous data
    :param data: data to separate
    :param labels: labels to separate by
    :return: normal and anomalous data
    """
    normal_data = data[labels]
    anomalous_data = data[~labels]

    return normal_data, anomalous_data


def preprocess_data(
    file_path: str, test_size: float = 0.2, random_state: int = 21
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Preprocess data by loading, splitting, normalising and separating by labels
    :param file_path: the path to the csv file
    :param test_size: the size of the test set
    :param random_state: the random state to use for splitting the data
    :return: normal train data, anomalous train data, normal test data, anomalous test data
    """
    # Load data
    data, labels = load_data(file_path)

    # Split data into train and test sets
    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labels, test_size=test_size, random_state=random_state
    )

    # Normalise data
    normalised_train_data, normalised_test_data = normalise_data(
        train_data=train_data, test_data=test_data
    )

    # Separate data by labels
    normal_train_data, anomalous_train_data = separate_data_by_labels(
        normalised_train_data, train_labels
    )
    normal_test_data, anomalous_test_data = separate_data_by_labels(
        normalised_test_data, test_labels
    )

    return (
        normal_train_data,
        anomalous_train_data,
        normal_test_data,
        anomalous_test_data,
    )
