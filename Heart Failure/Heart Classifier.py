# Imports
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Softmax
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Get dataset
data = pd.read_csv('Heart Failure Clinical Records.csv')
df = pd.DataFrame(data)

# Remove the 'DEATH_EVENT' value from the dataset for prediction
data.drop(['DEATH_EVENT', 'time'], axis = 1, inplace = True)
df2 = pd.DataFrame(data)

X = []
Y = []

# Add the necessary components to X and Y
for row in range(df2.shape[0]):
  rows = []
  for point in range(11):
    rows.append(df2.iloc[row][point])
  X.append(rows)

for row in range(df.shape[0]):
  Y.append(df.iloc[row][-1])

# Divide the X and Y values
trainx, testx, trainy, testy = train_test_split(X, Y, random_state = 1) # Use trainx and trainy instead of X and Y below

# Get input shape
input_shape = len(X[0])

# Create Adam optimizer
opt = Adam(learning_rate = 0.001)

# Create model
model = Sequential()

# Add an initial batch norm layer so that all the values are in a reasonable range for the network to process
model.add(BatchNormalization())
model.add(Dense(512, input_shape = [input_shape])) # Input layer

# Hidden layers
model.add(Dense(256, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))


model.add(Dense(128, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Dense(128, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

# Output layer
model.add(Dense(1, activation = 'sigmoid'))

# Compile model
model.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['binary_accuracy'])
early_stopping = EarlyStopping(min_delta = 0.001, patience = 100, restore_best_weights = True)

# Fit model and store training history
history = model.fit(trainx, trainy, validation_data = (testx, testy), batch_size = 512, epochs = 700, callbacks = [early_stopping])
history_df = pd.DataFrame(history.history)

# Plot loss and validation loss
history_df.loc[0:, ['loss', 'val_loss']].plot()
history_df.loc[0:, ['binary_accuracy', 'val_binary_accuracy']].plot()

# Prediction vs. actual value (change the index to view a different input and output set)
index = 0
prediction = np.argmax(model.predict([testx[index]]))
print(f"\nModel's Prediction on a Sample Input: {prediction}") # 0 if the the model predicts the patient will die, 1 if the the model predicts the patient will live
print(f"Actual Label on the Same Input: {testy[index]}")

# Evaluate the model
test_loss, test_acc = model.evaluate(np.array(testx), np.array(testy), verbose = 0) # Change verbose to 1 or 2 for more information
print(f'\nTest accuracy: {test_acc * 100}%')
