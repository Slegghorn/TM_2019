import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation, Dropout, BatchNormalization
import numpy as np
import wandb
import os
from wandb.keras import WandbCallback
wandb.init(project="dog-and-cats")

x = np.load('x_data.npy')
y = np.load('y_data.npy')
x = x/255.0


model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape = x.shape[1:], activation = tf.nn.relu))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), activation = tf.nn.relu))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), activation = tf.nn.relu))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation = tf.nn.relu))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), activation = tf.nn.relu))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), activation = tf.nn.relu))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(64, activation = tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(2, activation = tf.nn.softmax))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.fit(x, y, batch_size = 64, validation_split = 0.1, epochs = 20, callbacks = [WandbCallback()])

model.save(os.path.join(wandb.run.dir, "model.h5"))
