## Anomaly Detection Autoencoder - Electrocardiogram Dataset 

# STILL IN PROGRESS

The Anomaly Detector Autoencoder is a machine learning model used to detect anomalies in ECG data.

## Background on Autoencoders

An autoencoder is essentailly a neural network trained to copy it's input to it's output. Internally, this model has a hidden layer which describes code used to represent an input. However, this copying is restricted to copy approximetly, forcing the model to learn a compressed representation of the input. This allows for the model to prioritise which aspects of the input data are most important.
The architecture of an autoencoder includes an encoder aspect, and a decoder aspect.
- The encoder aspect is responsible for compressing the input data into a latent-space representation.
- The decoder aspect is responsible for reconstructing the input data from the latent-space representation.

## Requirements
To use the Anomaly Detector Autoencoder, you need the following dependencies:

- TensorFlow (version 2.13.0rc0)
- NumPy (version 1.24.3)
- Matplotlib (version 3.7.1)
- Scikit-learn (version 1.2.2)


## Conclusion
The Anomaly Detector Autoencoder provides a tool for detecting anomalies in ECG data. By leveraging an autoencoder architecture, it can learn representations of normal ECG patterns and identify deviations from these patterns. With proper data preprocessing and model training, it can be used to improve anomaly detection and contribute to the field of healthcare.

Feel free to modify the content as per your requirements and add any additional sections that you find necessary. Constructive feedback is always welcome!


## On the to-do list:
- [ ] incorporate some MLOPS 
- [ ] research and implement a better model architecture
- [ ] add more documentation eg diagrams
- [ ] better visualisations of results
- [ ] add tests

