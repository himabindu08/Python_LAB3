import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras.datasets import mnist
from keras.utils import np_utils
from keras.callbacks import TensorBoard
import tensorflow as tf
import matplotlib as plt
from sklearn.model_selection import train_test_split

batch_size = 128
nb_classes = 1
nb_epoch = 5


#loading the dataset
dataset = pd.read_csv("housingdata.csv", header=None).values
 #print(dataset)
import numpy as np
X_train, X_test, Y_train, Y_test = train_test_split(dataset[:,0:13], dataset[:,13],
                                                    test_size=0.25, random_state=87)
##Data normalization
X_train = X_train.astype(np.float)
X_test = X_test.astype(np.float)
X_train /= 255
X_test /= 255
#Y_Train = np_utils.to_categorical(Y_train, nb_classes)
#Y_Test = np_utils.to_categorical(Y_test, nb_classes)

# Linear regression model
model = Sequential()
model.add(Dense(output_dim=1, input_shape=(13,), init='normal', activation='sigmoid'))
model.compile(optimizer='SGD', loss='mean_absolute_error')
model.summary()

# Train
tensorboard = TensorBoard(log_dir="logs/{}",histogram_freq=0, write_graph=True, write_images=True)
model.fit(X_train, Y_train, nb_epoch=nb_epoch, batch_size=batch_size,callbacks=[tensorboard])

# Evaluates
score= model.evaluate(X_test, Y_test, verbose=1)
print()
#predicting the accuracy of the m
print(score)
#print(acc)
