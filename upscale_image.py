import os
import h5py
import time
import keras
import itertools
import scipy.misc
from loss import *
import numpy as np
from sklearn.utils import shuffle
from keras import backend as K
from keras.engine.topology import Layer
from keras.models import Sequential, Model
from keras.models import model_from_json, load_model
from keras.utils.data_utils import get_file
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Dense, Dropout, Flatten, merge, Add, Input, Concatenate, Average
from keras.layers import Conv2D, MaxPooling2D, Permute, Activation, BatchNormalization
import json
from imgaug import augmenters as iaa
import random
keras.backend.set_image_dim_ordering('th')
import math
import argparse
from keras_model import depth_to_scale_th, depth_to_scale_tf
from keras_model import SubPixelUpscaling, Normalize, Denormalize
from keras_model import _residual_block, _inception_residual_block
from keras_model import sr_model_X2_1, sr_model_X3_1, sr_model_X4_1

def upscale_dir(load_path, save_path):
    start_time =  time.clock()
    num_imgs = 0
    for filename in os.listdir(load_path):
        num_imgs += 1
        test_im = [scipy.misc.imread(load_path + filename)]
        sample = en_net.predict(gen.predict([np.transpose(test_im, (0, 3, 1, 2))]))
        scipy.misc.imsave(save_path + filename, np.transpose(sample[0], (1, 2, 0)))
    print ('time_per_image', (time.clock() - start_time) / num_imgs)

def upscale_image(im_path, save_path):
    start_time =  time.clock()
    img = np.transpose([scipy.misc.imread(im_path)], (0, 3, 1, 2))
    uimg = en_net.predict(gen.predict(img))
    scipy.misc.imsave(save_path + im_path.split('/')[-1], np.transpose(uimg[0], (1, 2, 0)))
    print ('time_taken', (time.clock() - start_time))

def enhancement_model(img_height=None, img_width=None):
    ip = Input(shape=(3, img_width, img_height), name="X_input")   
    ip_norm = Normalize(name='input_norm')(ip)

    conv1   = Conv2D(128, (5, 5), padding='same', strides=(1,1), name='en_conv1')(ip_norm)
    conv1_b = Activation('relu', name='sr_conv1_activ')(conv1)

    conv2   = Conv2D(128, (5, 5), padding='same', strides=(1,1), name='en_conv2')(conv1_b)
    conv2_b = Activation('relu', name='en_conv2_activ')(conv2)

    conv3   = Conv2D(128, (5, 5), padding='same', strides=(1,1), name='en_conv3')(conv2_b)
    conv3_b = Activation('relu', name='en_conv3_activ')(conv3)

    conv4   = Conv2D(128, (5, 5), padding='same', strides=(1,1), name='en_conv4')(conv3_b)
    conv4_b = Activation('relu', name='en_conv4_activ')(conv4)

    tv_regularizer = TVRegularizer(img_width=im_size * scale, img_height=im_size * scale,
                                       weight=1e-3)

    conv5 = Conv2D(3, (5, 5), padding='same', strides=(1,1), name='en_conv5', activation='tanh', activity_regularizer=tv_regularizer)(conv4_b)

    out = Average()([ip_norm, conv5])
    out = Denormalize()(out)

    en_net = Model(ip, out)
    return en_net

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Specify scale and image path')
    parser.add_argument('--image_path', default=None, help='Specify the image path')
    parser.add_argument('--image_dir', default=None, help='Specify an input directory')
    parser.add_argument('--scale', type=int, default=2, help='Specify upscaling factor')
    parser.add_argument('--save_path', default='./', help='Specify output directory')
    args = parser.parse_args()

    img_height = None
    img_width = None
    pool_type = 0

    scale = args.scale
    im_path = args.image_path
    im_dir = args.image_dir
    save_path = args.save_path

    model_name = 'model_resnet_x' + str(scale) + '.h5'
    en_model_name = 'en_model_resnet_x' + str(scale) + '.h5'

    im_size = 96
    num_filters = 64

    model = {2: sr_model_X2_1, 3: sr_model_X3_1, 4: sr_model_X4_1}
    gen = model[scale]()
    gen.load_weights('weights/' + model_name)

    en_net = enhancement_model()
    en_net.load_weights('weights/' + en_model_name)

    if im_path:
        upscale_image(im_path, save_path)
    if im_dir:
        upscale_dir(im_dir, save_path)