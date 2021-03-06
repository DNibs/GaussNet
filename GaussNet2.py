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

# FLAGS
FLAG_load_prev_model = True #Checks and loads GaussNet2.hdf5
FLAG_plot_training = False #Plot the generated training and eval data
FLAG_plot_history = True #Plot history of loss/accuracy during training
FLAG_predict = True #Make predictions with model after training
FLAG_hidden_layer_1 = True #32 nodes
FLAG_hidden_layer_2 = True #4 nodes
FLAG_hidden_layer_3 = True #4 nodes
FLAG_hidden_layer_4 = True #4 nodes
FLAG_hidden_layer_5 = True #4 nodes
FLAG_hidden_layer_6 = False #1 nodes, chokepoint


# Variables
samples = 6000
epochs = 20

# Path directories
hdf5_dir = os.makedirs('hdf5', exist_ok=True)
hdf5_path = 'hdf5/GaussNet2.hdf5'
checkpoint_path = "checkpoints/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Initiate Tensorboard
#   To see tensorboard, open terminal in directory and input "tensorboard --logdir=logs/"
#   It will output a link to open with browser. Browser will see real-time data from training.
#   DON'T FORGET TO QUIT with CTRL+C when finished!
tensorboard = keras.callbacks.TensorBoard(log_dir='logs/{}'.format(time()))

# Create checkpoint callback
cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path,
                                              save_weights_only=True,
                                              verbose=1,
                                              period=5)

# Create Early Stopping callback
early_stop = keras.callbacks.EarlyStopping(monitor='val_acc',
                                           min_delta=0.001,
                                           patience=5,
                                           verbose=1,
                                           mode='max')

# Generate Data - eval samples = 1000, so overall samples must be significantly greater
[train_data, train_labels, eval_data, eval_labels, train_data_class,
 eval_data_class] = DG.data_gen_2(samples=samples)


# Build Network
def create_GaussNet2():
    model = keras.Sequential()
    model.add(keras.layers.Dense(4, input_dim=2, activation=tf.nn.relu))
    if FLAG_hidden_layer_1 == True:
        model.add(keras.layers.Dense(32, activation=tf.nn.relu))
    if FLAG_hidden_layer_2 == True:
        model.add(keras.layers.Dense(4, activation=tf.nn.relu))
    if FLAG_hidden_layer_3 == True:
        model.add(keras.layers.Dense(4, activation=tf.nn.relu))
    if FLAG_hidden_layer_4 == True:
        model.add(keras.layers.Dense(4, activation=tf.nn.relu))
    if FLAG_hidden_layer_5 == True:
        model.add(keras.layers.Dense(4, activation=tf.nn.relu))
    if FLAG_hidden_layer_6 == True:
        model.add(keras.layers.Dense(1, activation=tf.nn.relu))
    model.add(keras.layers.Dense(4, activation=tf.nn.softmax))

    model.summary()

    # Compile model
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])

    return model

CHECK_model_not_ready = True # Forces creation of new model unless file loaded
CHECK_new_model = True # Avoids plotting history if training doesn't occur

# Load/Create the model
if FLAG_load_prev_model == True:
    try:
        model = keras.models.load_model(filepath=hdf5_path, compile=True)
        model.summary()
        CHECK_model_not_ready = False
        CHECK_new_model = False
    except OSError:
        CHECK_model_not_ready = True
        print('Unable to load model - creating new model')


if CHECK_model_not_ready == True:
    # Build/Optimize model
    model = create_GaussNet2()
    model.summary()
    # Train model
    history = model.fit(train_data,
                        train_labels,
                        validation_data=(eval_data, eval_labels),
                        epochs=40,
                        verbose=1,
                        callbacks=[tensorboard, cp_callback, early_stop])
    # Save entire model
    model.save(filepath=hdf5_path, overwrite=True, include_optimizer=True)
    CHECK_new_model = True

# Evaluate Model
accuracy = model.evaluate(eval_data, eval_labels)
print(accuracy)


# Plot training data
if FLAG_plot_training == True:
    DP.data_plot_2(train_data_class, eval_data_class)

if FLAG_plot_history == True:
    if CHECK_new_model == True:
        DP.results_plot(history)
    else:
        print('No history to print')

# Make Predictions and plot
if FLAG_predict == True:
    predict_data = DG.data_gen_predict(samples=samples)
    prediction = model.predict_classes(predict_data)
    DP.predict_plot_2(predict_data, prediction)


