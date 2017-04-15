
import copy
import numpy as np

from keras import backend as K


def smooth_gan_labels(y):
    assert len(y.shape) == 2, "Needs to be a binary class"
    y = np.asarray(y, dtype='int')
    Y = np.zeros(y.shape, dtype='float32')

    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            if y[i, j] == 0:
                Y[i, j] = np.random.uniform(0.0, 0.3)
            else:
                Y[i, j] = np.random.uniform(0.7, 1.2)

    return Y