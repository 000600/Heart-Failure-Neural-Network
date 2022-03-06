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
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

data = pd.read_csv('Heart Failure Clinical Records.csv')
df = pd.DataFrame(data)

Y = []

data.drop(['DEATH_EVENT', 'time'], axis = 1, inplace = True)
df2 = pd.DataFrame(data)
X = []

for row in range(df2.shape[0]):
  rows = []
  for point in range(11):
    rows.append(df2.iloc[row][point])
  X.append(rows)

for row in range(df.shape[0]):
  Y.append(df.iloc[row][-1])

print(X)
print(Y)

trainx, testx, trainy, testy = train_test_split(X, Y, random_state = 1)
x = np.array(trainx)
y = np.array(trainy)

opt = Adam(learning_rate = 0.001)

model = Sequential()

model.add(tf.keras.layers.BatchNormalization())
model.add(Dense(512, input_shape = [11]))

model.add(Dense(256, activation = 'relu'))
model.add(tf.keras.layers.BatchNormalization())

model.add(Dense(512, activation = 'relu'))
model.add(tf.keras.layers.BatchNormalization())

model.add(Dense(256, activation = 'relu'))
model.add(tf.keras.layers.BatchNormalization())

model.add(Dense(2))
model.add(Softmax())

model.compile(optimizer = opt, loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True), metrics=['accuracy'])

early_stopping = EarlyStopping(min_delta = 0.001, patience = 5, restore_best_weights = True)

history = model.fit(X, Y, epochs = 100) # callbacks = [early_stopping]
history_df = pd.DataFrame(history.history)

history_df.head()
history_df.loc[0:, 'loss'].plot()

print(np.argmax(model.predict([testx[0]])))
print(testy[0])

test_loss, test_acc = model.evaluate(np.array(testx), np.array(testy), verbose=2)
print('Test accuracy:', test_acc)
