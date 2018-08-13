# Simple binary classification neural net
# Takes 2d distribution and classifies whether positive/negative along x-axis

# Libraries
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Project Files
import Data_Generator as DG #Generates and classifies data for testing and validation
import Data_Plotter as DP #Gives functions for plotting data

print(tf.__version__)


# Variables
samples = 10000
epochs = 5


# Build Network
model = keras.Sequential()
model.add(keras.layers.Dense(4, input_dim=2, activation='relu'))
model.add(keras.layers.Dense(8, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.summary()

# Compile model
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Generate Data - eval samples = 100, so overall samples must be significantly greater
[train_data, train_labels, eval_data, eval_labels, train_data_class_0, train_data_class_1,
 eval_data_class_0, eval_data_class_1] = DG.data_gen(samples=samples)

# Train model
history = model.fit(train_data,
                    train_labels,
                    validation_data=(eval_data, eval_labels),
                    epochs=epochs,
                    verbose=1)


# Evaluate Model
accuracy = model.evaluate(eval_data, eval_labels)
print(accuracy)

# Plot results
DP.data_plot(train_data_class_0, train_data_class_1, eval_data_class_0, eval_data_class_1)
DP.results_plot(history)


# Make Predictions and plot
predict_data = DG.data_gen_predict(samples=samples)
prediction = model.predict_classes(predict_data)
DP.predict_plot(predict_data, prediction)