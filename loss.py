import keras.backend as K

def modified_binary_crossentropy(target, output):
    #output = K.clip(output, _EPSILON, 1.0 - _EPSILON)
    #return -(target * output + (1.0 - target) * (1.0 - output))
    return K.mean(target*output)
