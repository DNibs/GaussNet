# Plots Data and results

# Libraries
import numpy as np
import matplotlib.pyplot as plt


# Plot the classified data
def data_plot(train_data_class_0, train_data_class_1, eval_data_class_0, eval_data_class_1):

    plt.figure()
    plt.scatter(train_data_class_0[:,0], train_data_class_0[:,1], marker='.', label='Class: 0')
    plt.scatter(train_data_class_1[:,0], train_data_class_1[:,1], marker='.', label='Class: 1')
    plt.ylim(-1, 1)
    plt.xlim(-1, 1)
    plt.axvline(x=0, color='k', linewidth='1')
    plt.title('(Close to continue)', fontsize=10)
    plt.suptitle('Training Data')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.draw()

    plt.figure()
    plt.scatter(eval_data_class_0[:,0], eval_data_class_0[:,1], marker='.', label='Class: 0')
    plt.scatter(eval_data_class_1[:,0], eval_data_class_1[:,1], marker='.', label='Class: 1')
    plt.ylim(-1, 1)
    plt.xlim(-1, 1)
    plt.axvline(x=0, color='k', linewidth='1')
    plt.title('(Close to continue)', fontsize=10)
    plt.suptitle('Eval Data')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.draw()

    plt.show()

    return

# Create graph of accuracy and loss over time
def results_plot(history):
    history_dict = history.history
    history_dict.keys()

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    # bo is for blue dot
    plt.figure()
    plt.plot(epochs, loss, 'bo', label = 'Training loss')
    # b is for solid blue line
    plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
    plt.suptitle('Training and validation loss')
    plt.title('(Close to continue)', fontsize=10)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.draw()


    #plt.clf() # clear figure
    acc_values = history_dict['acc']
    val_acc_values = history_dict['val_acc']

    plt.figure()
    plt.plot(epochs, acc, 'bo', label = 'Training acc')
    plt.plot(epochs, val_acc, 'b', label = 'Validation acc')
    plt.suptitle('Training and validation accuracy')
    plt.title('(Close to continue)', fontsize=10)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.draw()

    plt.show()
    return


# Plots results from prediction.
def predict_plot(predict_data, prediction):

    # Split predict_data into two lists based on results of prediction
    predict_data_class_0 = []
    predict_data_class_1 = []
    for i in range(0, len(prediction) ):
        if prediction[i] == 0:
            predict_data_class_0.append(predict_data[i,:])
        else:
            predict_data_class_1.append(predict_data[i,:])

    # Turn lists into arrays for plotting
    predict_data_class_0 = np.asarray(predict_data_class_0)
    predict_data_class_1 = np.asarray(predict_data_class_1)

    # Plot figure
    plt.figure()
    plt.scatter(predict_data_class_0[:,0], predict_data_class_0[:,1], marker='.', label='Class: 0')
    plt.scatter(predict_data_class_1[:,0], predict_data_class_1[:,1], marker='.', label='Class: 1')
    plt.ylim(-1, 1)
    plt.xlim(-1, 1)
    plt.axvline(x=0, color='k', linewidth='1')
    plt.title('(Close to conclude)', fontsize=10)
    plt.suptitle('Prediction Results')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.draw()

    plt.show()

    return


# Plot the classified data
def data_plot_2(train_data_class, eval_data_class):

    plt.figure()
    for i in range(0,4):
        plt.scatter(train_data_class[i][:,0],
                    train_data_class[i][:,1], marker='.', label='Class: {}'.format(i))
    plt.ylim(-1, 1)
    plt.xlim(-1, 1)
    plt.axvline(x=0, color='k', linewidth='1')
    plt.axhline(y=0, color='k', linewidth='1')
    plt.title('(Close to continue)', fontsize=10)
    plt.suptitle('Training Data')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.draw()

    plt.figure()
    for i in range(0,4):
        plt.scatter(eval_data_class[i][:,0],
                    eval_data_class[i][:,1], marker='.', label='Class: {}'.format(i))
    plt.ylim(-1, 1)
    plt.xlim(-1, 1)
    plt.axvline(x=0, color='k', linewidth='1')
    plt.axhline(y=0, color='k', linewidth='1')
    plt.title('(Close to continue)', fontsize=10)
    plt.suptitle('Eval Data')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.draw()

    plt.show()

    return


# Plots results from prediction for GaussNet2.
def predict_plot_2(predict_data, prediction):

    # Split predict_data into two lists based on results of prediction
    count = np.array([0,0,0,0])
    predict_data_class = [[] for i in range(0,4)]
    for i in range(0, len(prediction) ):
        if prediction[i] == 0:
            predict_data_class[0].append(predict_data[i,:])
            count[0] = count[0] + 1
        elif prediction[i] == 1:
            predict_data_class[1].append(predict_data[i,:])
            count[1] = count[1] + 1
        elif prediction[i] == 2:
            predict_data_class[2].append(predict_data[i,:])
            count[2] = count[2] + 1
        else:
            predict_data_class[3].append(predict_data[i, :])
            count[3] = count[3] + 1


    # Turn lists into arrays for plotting
    for i in range(0,4):
        if count[i] == 0:
            predict_data_class[i] = [[0,0],[0,1]]
        predict_data_class[i] = np.asarray(predict_data_class[i])
    predict_data_class = np.asarray(predict_data_class)

    # Plot figure
    plt.figure()
    for i in range(0,4):
        plt.scatter(predict_data_class[i][:,0],
                    predict_data_class[i][:,1], marker='.', label='Class: {}'.format(i))
    plt.ylim(-1, 1)
    plt.xlim(-1, 1)
    plt.axvline(x=0, color='k', linewidth='1')
    plt.axhline(y=0, color='k', linewidth='1')
    plt.title('(Close to conclude)', fontsize=10)
    plt.suptitle('Prediction Results')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.draw()

    plt.show()

    return
