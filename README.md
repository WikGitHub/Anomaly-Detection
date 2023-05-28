## Anomaly Detection Autoencoder - Electrocardiogram Dataset 

# STILL IN PROGRESS

The Anomaly Detector Autoencoder is a machine learning model used to detect anomalies in ECG data. It utilises an autoencoder architecture consisting of an encoder and a decoder.

## Requirements
To use the Anomaly Detector Autoencoder, you need the following dependencies:

TensorFlow (version 2.13.0rc0)
NumPy (version 1.24.3)
Matplotlib (version 3.7.1)
Scikit-learn (version 1.2.2)

## Usage

### Model Definition
The AnomalyDetector class represents the autoencoder model. It is defined in the anomaly_detector.py file. The autoencoder architecture consists of an encoder and a decoder, each composed of several dense layers with specific activation functions.

### Data Preprocessing
The preprocess_data function in the data_preprocessing.py file is used to preprocess the ECG data. It loads the data from a CSV file, performs normalisation, and separates the data into normal and anomalous.

### Training and Evaluation
The train_autoencoder function trains the autoencoder model on the provided training data. It takes the autoencoder model, training data, and optional parameters such as the number of epochs and validation data. The model is compiled with the Adam optimiser and Mean Absolute Error (MAE) loss function.

The evaluate_autoencoder function evaluates the trained autoencoder model on the provided test data. It calculates the loss value and logs the evaluation result.

### Productionising
The productionise_autoencoder function is used to streamline the training and evaluation process. It calls the train_autoencoder function to train the model on the training data and then evaluates the model using the evaluate_autoencoder function on the test data.

## Example Usage
To use the Anomaly Detector Autoencoder, follow these steps:

Preprocess the data by calling the preprocess_data function to obtain the normal train data, anomalous train data, normal test data, and anomalous test data.

Create an instance of the AnomalyDetector model.

Call the productionise_autoencoder function, passing the autoencoder model and the preprocessed data as arguments.


## Conclusion
The Anomaly Detector Autoencoder provides a tool for detecting anomalies in ECG data. By leveraging an autoencoder architecture, it can learn representations of normal ECG patterns and identify deviations from these patterns. With proper data preprocessing and model training, it can be used to improve anomaly detection and contribute to the field of healthcare.

Feel free to modify the content as per your requirements and add any additional sections that you find necessary. Constructive feedback is always welcome!


## On the to-do list:
- [ ] incorporate some MLOPS 
- [ ] research and implement a better model architecture

