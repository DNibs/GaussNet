# Simple classification neural net
# Takes 2d distribution and classifies the four quadrants

# Libraries
from __future__ import absolute_import, division, print_function #allows saving parameters
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from time import time #allows logging
import os #allows saving model parameters

# Project Files
import Data_Generator as DG #Generates and classifies data for testing and validation
import Data_Plotter as DP #Gives functions for plotting data

print(tf.__version__)


# Variables
samples = 6000
epochs = 20
FLAG_hidden_layer_1 = 1 #32 nodes
FLAG_hidden_layer_2 = 0 #4 nodes
FLAG_hidden_layer_3 = 0 #4 nodes
FLAG_hidden_layer_4 = 0 #4 nodes
FLAG_hidden_layer_5 = 0 #4 nodes
FLAG_hidden_layer_6 = 0 #1 nodes, chokepoint
FLAG_plot_training = 0
FLAG_plot_history = 0

# Build Network
model = keras.Sequential()
model.add(keras.layers.Dense(4, input_dim=2, activation=tf.nn.relu))
if FLAG_hidden_layer_1 == 1:
    model.add(keras.layers.Dense(32, activation=tf.nn.relu))
if FLAG_hidden_layer_2 == 1:
    model.add(keras.layers.Dense(4, activation=tf.nn.relu))
if FLAG_hidden_layer_3 == 1:
    model.add(keras.layers.Dense(4, activation=tf.nn.relu))
if FLAG_hidden_layer_4 == 1:
    model.add(keras.layers.Dense(4, activation=tf.nn.relu))
if FLAG_hidden_layer_5 == 1:
    model.add(keras.layers.Dense(4, activation=tf.nn.relu))
if FLAG_hidden_layer_6 == 1:
    model.add(keras.layers.Dense(1, activation=tf.nn.relu))
model.add(keras.layers.Dense(4, activation=tf.nn.softmax))

model.summary()

# Compile model
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Initiate Tensorboard
#   To see tensorboard, open terminal in directory and input "tensorboard --logdir=logs/"
#   It will output a link to open with browser. Browser will see real-time data from training.
#   DON'T FORGET TO QUIT with CTRL+C when finished!
tensorboard = keras.callbacks.TensorBoard(log_dir='logs/{}'.format(time()))

# Create checkpoint callback (to save/load parameters)
checkpoint_path = "checkpoints/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path,
                                              save_weights_only=True,
                                              verbose=1,
                                              period=5)


# Generate Data - eval samples = 1000, so overall samples must be significantly greater
[train_data, train_labels, eval_data, eval_labels, train_data_class,
 eval_data_class] = DG.data_gen_2(samples=samples)


# Train model
history = model.fit(train_data,
                    train_labels,
                    validation_data=(eval_data, eval_labels),
                    epochs=epochs,
                    verbose=1,
                    callbacks=[tensorboard, cp_callback])


# Evaluate Model
accuracy = model.evaluate(eval_data, eval_labels)
print(accuracy)


# Plot results
if FLAG_plot_training == 1:
    DP.data_plot_2(train_data_class, eval_data_class)
if FLAG_plot_history == 1:
    DP.results_plot(history)

# Make Predictions and plot
predict_data = DG.data_gen_predict(samples=samples)
prediction = model.predict_classes(predict_data)
DP.predict_plot_2(predict_data, prediction)


# Save entire model
hdf5_path = 'hdf5/GaussNet2.hdf5'
hdf5_dir = os.makedirs('hdf5', exist_ok=True)
model.save(filepath= hdf5_path)