# Generates and classifies Data for the model
# Data is split into training and evaluation

import numpy as np
import matplotlib.pyplot as plt


# Create 2d Gaussian Distribution

def data_gen(samples):
    # Parameters
    mean = [0, 0]
    cov = [[1, 0], [0,1]]

    x,y = np.random.multivariate_normal(mean=mean, cov=cov, size=samples).T

    # make 2d array instead of two lists
    train_data = np.asarray(list(zip(x[:samples-100],y[:samples-100])))
    eval_data = np.asarray(list(zip(x[samples-100:],y[samples-100:])))

    # Normalize to [-1, 1]
    train_max = np.amax( [abs(i) for i in train_data], axis=0)
    train_data[:,0] = train_data[:,0]/train_max[0]
    train_data[:,1] = train_data[:,1]/train_max[1]

    eval_max = np.amax( [abs(i) for i in eval_data], axis=0)
    eval_data[:,0] = eval_data[:,0]/eval_max[0]
    eval_data[:,1] = eval_data[:,1]/eval_max[1]


    # Classify distribution
    #   Labels are assigned: if x<0, 0; if x>0, 1
    train_labels = []
    train_data_class_0 = []
    train_data_class_1 = []
    eval_labels = []
    eval_data_class_0 = []
    eval_data_class_1 = []

    for i in range(0,samples-100):
        if train_data[i,0] < 0:
            train_labels.append(0)
            train_data_class_0.append(train_data[i,:])
        else:
            train_labels.append(1)
            train_data_class_1.append(train_data[i,:])

    for i in range(0,100):
        if eval_data[i,0] < 0:
            eval_labels.append(0)
            eval_data_class_0.append(eval_data[i,:])
        else:
            eval_labels.append(1)
            eval_data_class_1.append(eval_data[i,:])

    # Convert lists to arrays
    train_data_class_0 = np.asarray(train_data_class_0)
    train_data_class_1 = np.asarray(train_data_class_1)

    eval_data_class_0 = np.asarray(eval_data_class_0)
    eval_data_class_1 = np.asarray(eval_data_class_1)
    return [train_data,
           train_labels,
           eval_data,
           eval_labels,
           train_data_class_0,
           train_data_class_1,
           eval_data_class_0,
           eval_data_class_1]

def data_gen_predict(samples):
    # Parameters
    mean = [0, 0]
    cov = [[1, 0], [0,1]]

    x,y = np.random.multivariate_normal(mean=mean, cov=cov, size=samples).T
    # Transform data into array instead of two lists
    predict_data = np.asarray(list(zip(x[:samples],y[:samples])))

    # Normalize to [-1, 1]
    predict_max = np.amax( [abs(i) for i in predict_data], axis=0)
    predict_data[:,0] = predict_data[:,0]/predict_max[0]
    predict_data[:,1] = predict_data[:,1]/predict_max[1]

    return predict_data

def data_gen_2(samples):
    # Parameters
    mean = [0, 0]
    cov = [[1, 0], [0,1]]

    x,y = np.random.multivariate_normal(mean=mean, cov=cov, size=samples).T

    # make 2d array instead of two lists
    train_data = np.asarray(list(zip(x[:samples-1000],y[:samples-1000])))
    eval_data = np.asarray(list(zip(x[samples-1000:],y[samples-1000:])))

    # Normalize to [-1, 1]
    train_max = np.amax( [abs(i) for i in train_data], axis=0)
    train_data[:,0] = train_data[:,0]/train_max[0]
    train_data[:,1] = train_data[:,1]/train_max[1]

    eval_max = np.amax( [abs(i) for i in eval_data], axis=0)
    eval_data[:,0] = eval_data[:,0]/eval_max[0]
    eval_data[:,1] = eval_data[:,1]/eval_max[1]


    # Classify distribution
    #   Labels are assigned: if x<0, 0; if x>0, 1
    train_labels = []
    train_data_class = [[] for i in range(0,4)]
    eval_labels = []
    eval_data_class = [[] for i in range(0,4)]

    for i in range(0,samples-1000):
        if (train_data[i,0] < 0 and train_data[i,1] < 0):
            train_labels.append(0)
            train_data_class[0].append(train_data[i,:])
        elif ((train_data[i,0] < 0) and (train_data[i,1] > 0)):
            train_labels.append(1)
            train_data_class[1].append(train_data[i,:])
        elif (train_data[i,0] > 0 and train_data[i,1] < 0):
            train_labels.append(2)
            train_data_class[2].append(train_data[i,:])
        else:
            train_labels.append(3)
            train_data_class[3].append(train_data[i,:])


    for i in range(0,1000):
        if (eval_data[i,0] < 0 and eval_data[i,1] < 0):
            eval_labels.append(0)
            eval_data_class[0].append(eval_data[i,:])
        elif (eval_data[i,0] < 0 and eval_data[i,1] > 0):
            eval_labels.append(1)
            eval_data_class[1].append(eval_data[i,:])
        elif (eval_data[i,0] > 0 and eval_data[i,1] < 0):
            eval_labels.append(2)
            eval_data_class[2].append(eval_data[i,:])
        else:
            eval_labels.append(3)
            eval_data_class[3].append(eval_data[i,:])


    # Convert lists to array
    for i in range(0,4):
        train_data_class[i] = np.asarray(train_data_class[i])
        eval_data_class[i] = np.asarray(eval_data_class[i])
    train_data_class = np.asarray(train_data_class)
    eval_data_class = np.asarray(eval_data_class)

    return [train_data,
           train_labels,
           eval_data,
           eval_labels,
           train_data_class,
           eval_data_class]
