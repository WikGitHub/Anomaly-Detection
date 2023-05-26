## Anomaly Detection Autoencoder - electrocardiogram dataset 

Using errors to detect anomalies in time series data.
The model is trained on normal ECGs and then tested on anomalous ECGs. The idea is that the autoencoder will learn to encode normal ECGs and then when it is presented with an anomalous ECG it will not be able to encode it as well and the error will be higher. This way we can detect anomalies in the data.



