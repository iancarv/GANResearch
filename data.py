import scipy.io
from keras.datasets import mnist
import numpy as np
from skimage.transform import resize

def load_mnist_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = np.expand_dims(X_train, axis=1)

    X_test = (X_test.astype(np.float32) - 127.5) / 127.5
    X_test = np.expand_dims(X_test, axis=1)
    return (X_train, y_train, X_test, y_test)

def load_tmi_data():
    # Load the dataset
    dataset = scipy.io.loadmat('data/training.mat')
    # Split into train and test. Values are in range [0..1] as float64
    X_train = np.transpose(dataset['train_x'], (3, 2, 1, 0))
    print(X_train.shape)
    y_train = list(dataset['train_y'][0])
    
    X_test = np.transpose(dataset['test_x'], (3, 2, 1, 0))
    y_test = list(dataset['test_y'][0])
    
    # Change shape and range. 
    y_train = np.asarray(y_train).reshape(-1, 1)
    y_test = np.asarray(y_test).reshape(-1, 1)

#   1-> 0 : Non-nucleus. 2 -> 1: Nucleus
    y_test -= 1
    y_train -= 1

    # Resize to 32x32
    X_train_resized = np.empty([X_train.shape[0], X_train.shape[3], 28, 28])
    for i in range(X_train.shape[0]):
        X_train_resized[i] = resize(X_train[i], (X_train.shape[3], 28, 28), mode='reflect')

    X_test_resized = np.empty([X_test.shape[0], X_test.shape[3], 28, 28])
    for i in range(X_test.shape[0]):
        X_test_resized[i] = resize(X_test[i], (X_train.shape[3], 28, 28), mode='reflect')
    
    X_train_resized = (2 * X_train_resized) - 1
    X_test_resized = (2 * X_test_resized) - 1

    return X_train_resized, y_train, X_test_resized, y_test
