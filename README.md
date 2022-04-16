# Heart Failure 

## The Neural Network

The network is densely connected and  predicts the probability of one's heart failing. Since the dataset is categorical (the model predicts either 0 or 1 for a death event), the model uses a sparse categorical crossentropy loss function. The model uses a standard ReLU activation function and Adam optimizer (with a learning rate of 0.001).

Feel free to further tune the hyperparameters or build upon the model!

## The Dataset
The dataset can be found at this link: https://www.kaggle.com/andrewmvd/heart-failure-clinical-data. Credit for the dataset collection goes to **Larxel** on *Kaggle*. It describes the death event (0 or 1) of a patient based on 11 factors, inluding:
- Age
- Anaemia
- Creatinine Phosphokinase
- Diabetes
- Blood Pressure
- Platelets

## Libraries
This neural network was created with the help of the Tensorflow and Scikit-Learn libraries.
- Tensorflow's Website: https://www.tensorflow.org/
- Tensorflow Installation Instructions: https://www.tensorflow.org/install
- Scikit Learn's Website: https://scikit-learn.org/stable/
- Scikit Learn's Installation Instructions: https://scikit-learn.org/stable/install.html
