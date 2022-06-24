# Heart Failure Neural Network

## The Neural Network

This network predicts whether or not a patient will die as a result of heart failure based on clinical heart records. The network is fully densely connected and, since the dataset is binary and categorical (the model predicts either 0 for patient survival or 1 for patient death), the model uses a binary crossentropy loss function and has 1 output neuron. The model uses a standard Adam optimizer with a learning rate of 0.001 and dropout layers to prevent overfitting. The model's architecture contains:
- 1 Batch Normalization layer
- 1 Input layer (with 512 input neurons)
- 3 Hidden layers (2 with 256 neurons and 1 with 512; each with a ReLU activation function)
- 3 Batch Normalization layers (one after each hidden layer)
- 4 Dropout layers (one after each hidden layer and the input layer)
- 1 Output Layer (with 1 output neuron and a sigmoid activation function)

Feel free to further tune the hyperparameters or build upon the model!

## The Dataset
The dataset can be found at this link: https://www.kaggle.com/andrewmvd/heart-failure-clinical-data. Credit for the dataset collection goes to **Nayan Sakhiya**, **Durgance Gaur**, **Sanchita Karmakar**, and others on *Kaggle*. It describes the death event (0 or 1) of a patient based on 11 factors, inluding:
- Age
- Anaemia
- Creatinine Phosphokinase
- Diabetes
- Blood Pressure
- Platelets

Although the raw data includes some bias (it is more representative of survival scenarios than death scenarios), this is solved in the model through SMOTE, which will oversample the death scenario in order to provide more equal data representation.

## Libraries
This neural network was created with the help of the Tensorflow and Scikit-Learn libraries.
- Tensorflow's Website: https://www.tensorflow.org/
- Tensorflow Installation Instructions: https://www.tensorflow.org/install
- Scikit Learn's Website: https://scikit-learn.org/stable/
- Scikit Learn's Installation Instructions: https://scikit-learn.org/stable/install.html
