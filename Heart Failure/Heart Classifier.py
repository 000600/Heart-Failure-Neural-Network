# Imports
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Softmax
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
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
model.add(tf.keras.layers.BatchNormalization())
model.add(Dense(512, input_shape = [input_shape])) # Input layer

# Hidden layers
model.add(Dense(256, activation = 'relu'))
model.add(tf.keras.layers.BatchNormalization())

model.add(Dense(512, activation = 'relu'))
model.add(tf.keras.layers.BatchNormalization())

model.add(Dense(256, activation = 'relu'))
model.add(tf.keras.layers.BatchNormalization())

# Output layer
model.add(Dense(2))
model.add(Softmax())

# Compile model
model.compile(optimizer = opt, loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True), metrics=['accuracy'])
early_stopping = EarlyStopping(min_delta = 0.001, patience = 5, restore_best_weights = True)

# Fit model and store training history
history = model.fit(X, Y, epochs = 100) # To add callbacks, add the following as a parameter: callbacks = [early_stopping]
history_df = pd.DataFrame(history.history)

# View the model's loss
history_df.head()
history_df.loc[0:, 'loss'].plot()

# Prediction vs. actual value (change the index to view a different input and output set
print(np.argmax(model.predict([testx[0]])))
print(testy[0])

# Evaluate the model
test_loss, test_acc = model.evaluate(np.array(testx), np.array(testy), verbose = 0) # Change verbose to 1 or 2 for more information
print('Test accuracy:', test_acc)
