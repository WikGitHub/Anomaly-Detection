import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from typing import Tuple, Union


def load_data(file_path: str) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Load data from csv file and return data and labels as tensors
    :param file_path: path to csv file
    :return: data and labels as tensors
    """
    dataframe = pd.read_csv(file_path)
    data = dataframe.iloc[:, :-1].values
    labels = dataframe.iloc[:, -1].astype(bool).values
    return data, labels


def normalise_data(data: tf.Tensor) -> tf.Tensor:
    """
    Normalise data to values between 0 and 1 using min-max normalisation
    :param data: data to normalise
    :return: normalised data
    """
    min_val = tf.reduce_min(data)
    max_val = tf.reduce_max(data)
    normalised_data = tf.cast((data - min_val) / (max_val - min_val), tf.float32)
    return normalised_data


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
    normalised_train_data = normalise_data(train_data)
    normalised_test_data = normalise_data(test_data)

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
