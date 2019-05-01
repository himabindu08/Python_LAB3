
import datetime as dt
import glob
import itertools
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from keras import models, layers, optimizers
from keras.applications import Xception
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
from sklearn.metrics import confusion_matrix
weights = Path('xception_weights_tf_dim_ordering_tf_kernels_notop.h5')
train_dir = Path('training/')
test_dir = Path('validation/')
cols = ['Label','Latin Name', 'Common Name','Train Images', 'Validation Images']
labels = pd.read_csv("monkey_labels.txt", names=cols, skiprows=1)
labels = labels['Common Name']
height=150
width=150
channels=3
batch_size=32
seed=1337


# Training generator
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(height,width),
                                                    batch_size=batch_size,
                                                    seed=seed,
                                                    class_mode='categorical')

# Test generator
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(test_dir,
                                                  target_size=(height,width),
                                                  batch_size=batch_size,
                                                  seed=seed,
                                                  class_mode='categorical')
base_model = Xception(weights=weights,
                      include_top=False,
                      input_shape=(height, width, channels))
base_model.summary()


def extract_features(sample_count, datagen):
    start = dt.datetime.now()
    features = np.zeros(shape=(sample_count, 5, 5, 2048))
    labels = np.zeros(shape=(sample_count, 10))
    generator = datagen
    i = 0
    for inputs_batch, labels_batch in generator:
        stop = dt.datetime.now()
        time = (stop - start).seconds
        print('\r',
              'Extracting features from batch', str(i + 1), '/', len(datagen),
              '-- run time:', time, 'seconds',
              end='')

        features_batch = base_model.predict(inputs_batch)
        features[i * batch_size: (i + 1) * batch_size] = features_batch
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1

        if i * batch_size >= sample_count:
            break

    print("\n")

    return features, labels
train_features, train_labels = extract_features(1097, train_generator)
test_features, test_labels = extract_features(272, test_generator)
flat_dim = 5 * 5 * 2048
train_features = np.reshape(train_features, (1097, flat_dim))
test_features = np.reshape(test_features, (272, flat_dim))
reduce_learning_rate = ReduceLROnPlateau(monitor='loss',
                                         factor=0.1,
                                         patience=2,
                                         cooldown=2,
                                         min_lr=0.00001,
                                         verbose=1)

callbacks = [reduce_learning_rate]
model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_dim=flat_dim))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['acc'])
model.summary()
history = model.fit(train_features,
                    train_labels,
                    epochs=30,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_split=0.1,
                    callbacks=callbacks)
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.title('Training and validation accuracy')
plt.plot(epochs, acc, 'red', label='Training acc')
plt.plot(epochs, val_acc, 'blue', label='Validation acc')
plt.legend()

plt.figure()
plt.title('Training and validation loss')
plt.plot(epochs, loss, 'red', label='Training loss')
plt.plot(epochs, val_loss, 'blue', label='Validation loss')

plt.legend()

plt.show()


