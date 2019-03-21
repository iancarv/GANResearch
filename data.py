import scipy.io
# from keras.datasets import mnist
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
    y_train = np.asarray(y_train).reshape(-1)
    y_test = np.asarray(y_test).reshape(-1)

#   1-> 0 : Non-nucleus. 2 -> 1: Nucleus
    y_test -= 1
    y_train -= 1

    # Resize to 32x32
    X_train_resized = np.empty([X_train.shape[0], X_train.shape[1], 28, 28])
    for i in range(X_train.shape[0]):
        X_train_resized[i] = resize(X_train[i], (X_train.shape[1], 28, 28), mode='reflect')

    X_test_resized = np.empty([X_test.shape[0], X_test.shape[1], 28, 28])
    for i in range(X_test.shape[0]):
        X_test_resized[i] = resize(X_test[i], (X_train.shape[1], 28, 28), mode='reflect')
    
    X_train_resized = (2 * X_train_resized) - 1
    X_test_resized = (2 * X_test_resized) - 1
    print(X_train_resized.shape)
    print(np.max(X_train_resized))
    print(np.min(X_train_resized))
    return X_train_resized, y_train, X_test_resized, y_test


def nuclei_position(cell):
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    ret,gray = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(gray,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    selected = []
    not_found = []
    dSquared = 17*17
    for c in contours[1:]:
        M = cv2.moments(c)
        cY = int(M["m10"] / M["m00"])
        cX = int(M["m01"] / M["m00"])
        yield (cY-17,cX-17,cY-17,cy+17)
      

def create_nuclei_data():
    pickle_in = open('data/out',"rb")
    m = pickle.load(pickle_in)
    all_images = np.array(list(m.keys())[:10])
    for key in all_images:
        d = m[key]
        cnt = 0
        crop = d['crop']
        cell = d['cell']
        if crop.shape[0] != 400 or crop.shape[1] != 400:
          print('Menas')
          continue
        for w in nuclei_position:
            cv2.imwrite('%s_%d.png' % (key, cnt), crop[w[0]:w[2], w[1]:w[3]])
            cnt += 1

    return X_test, y_test

