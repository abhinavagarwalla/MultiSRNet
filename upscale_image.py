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

''' Theano Backend function '''
def depth_to_scale_th(input, scale, channels):
    ''' Uses phase shift algorithm [1] to convert channels/depth for spacial resolution '''
    import theano.tensor as T

    b, k, row, col = input.shape
    output_shape = (b, channels, row * scale, col * scale)

    out = T.zeros(output_shape)
    r = scale

    for y, x in itertools.product(range(scale), repeat=2):
        out = T.inc_subtensor(out[:, :, y::r, x::r], input[:, r * y + x :: r * r, :, :])

    return out

''' Tensorflow Backend Function '''
def depth_to_scale_tf(input, scale, channels):
    try:
        import tensorflow as tf
    except ImportError:
        print("Could not import Tensorflow for depth_to_scale operation. Please install Tensorflow or switch to Theano backend")
        exit()

    def _phase_shift(I, r):
        ''' Function copied as is from https://github.com/Tetrachrome/subpixel/blob/master/subpixel.py'''

        bsize, a, b, c = I.get_shape().as_list()
        bsize = tf.shape(I)[0]  # Handling Dimension(None) type for undefined batch dim
        X = tf.reshape(I, (bsize, a, b, r, r))
        X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
        X = tf.split(1, a, X)  # a, [bsize, b, r, r]
        X = tf.concat(2, [tf.squeeze(x) for x in X])  # bsize, b, a*r, r
        X = tf.split(1, b, X)  # b, [bsize, a*r, r]
        X = tf.concat(2, [tf.squeeze(x) for x in X])  # bsize, a*r, b*r
        return tf.reshape(X, (bsize, a * r, b * r, 1))

    if channels > 1:
        Xc = tf.split(3, 3, input)
        X = tf.concat(3, [_phase_shift(x, scale) for x in Xc])
    else:
        X = _phase_shift(input, scale)
    return X

''' Implementation is incomplete. Use lambda layer for now. '''
class SubPixelUpscaling(Layer):

    def __init__(self, r, channels, **kwargs):
        super(SubPixelUpscaling, self).__init__(**kwargs)

        self.r = r
        self.channels = channels

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        if K.backend() == "theano":
            y = depth_to_scale_th(x, self.r, self.channels)
        else:
            y = depth_to_scale_tf(x, self.r, self.channels)
        return y

    def compute_output_shape(self, input_shape):
        if K.image_dim_ordering() == "th":
            b, k, r, c = input_shape
            if r is not None:
                return (b, self.channels, r * self.r, c * self.r)
            else:
                return (b, self.channels, r, c)
        else:
            b, r, c, k = input_shape
            return (b, r * self.r, c * self.r, self.channels)

class Denormalize(Layer):
    '''
    Custom layer to denormalize the final Convolution layer activations (tanh)

    Since tanh scales the output to the range (-1, 1), we add 1 to bring it to the
    range (0, 2). We then multiply it by 127.5 to scale the values to the range (0, 255)
    '''

    def __init__(self, **kwargs):
        super(Denormalize, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        '''
        Scales the tanh output activations from previous layer (-1, 1) to the
        range (0, 255)
        '''

        return (x + 1) * 127.5

    def compute_output_shape(self, input_shape):
        return input_shape

class Normalize(Layer):
    '''
    Custom layer to normalize the image to range [-1, 1].
    '''

    def __init__(self, **kwargs):
        super(Normalize, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        '''
        Since the input image has range [0, 255]
        '''

        return (x / 127.5) - 1.0

    def compute_output_shape(self, input_shape):
        return input_shape

def PSNRLoss(y_true, y_pred):
    """
    PSNR is Peek Signal to Noise Ratio, which is similar to mean squared error.
    It can be calculated as
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE)
    When providing an unscaled input, MAXp = 255. Therefore 20 * log10(255)== 48.1308036087.
    However, since we are scaling our input, MAXp = 1. Therefore 20 * log10(1) = 0.
    Thus we remove that component completely and only compute the remaining MSE component.
    """
    y_true = y_true / 255.0
    y_pred = y_pred / 255.0
    return -10. * np.log10(K.mean(K.square(y_pred - y_true)))

def _residual_block(ip, id, axis_b):
    init = ip

    x = Conv2D(num_filters, (3, 3), padding='same', name='rb_conv_' + str(id) + '_1')(ip)
    x = BatchNormalization(axis=axis_b, name="rb_batchnorm_" + str(id) + "_1")(x)
    x = Activation('relu', name="rb_activation_" + str(id) + "_1")(x)

    x = Conv2D(num_filters, (3, 3), padding='same', name='rb_conv_' + str(id) + '_2')(x)
    x = BatchNormalization(axis=axis_b, name="rb_batchnorm_" + str(id) + "_2")(x)

    m = Add()([x, init])
    return m

def _inception_residual_block(ip, id, axis_b):
    init = ip

    x1 = Conv2D(num_filters, (1, 1), padding='same', name='rb_conv_' + str(id) + '_11')(ip)
    x1 = BatchNormalization(axis=axis_b, name="rb_batchnorm_" + str(id) + "_11")(x1)
    x1 = Activation('relu', name="rb_activation_" + str(id) + "_11")(x1)

    x2 = Conv2D(num_filters, (3, 3), padding='same', name='rb_conv_' + str(id) + '_12')(ip)
    x2 = BatchNormalization(axis=axis_b, name="rb_batchnorm_" + str(id) + "_12")(x2)
    x2 = Activation('relu', name="rb_activation_" + str(id) + "_12")(x2)

    x3 = Conv2D(num_filters, (5, 5), padding='same', name='rb_conv_' + str(id) + '_13')(ip)
    x3 = BatchNormalization(axis=axis_b, name="rb_batchnorm_" + str(id) + "_13")(x3)
    x3 = Activation('relu', name="rb_activation_" + str(id) + "_13")(x3)

    x = Concatenate(axis=1)([x1, x2, x3])

    x = Conv2D(num_filters, (3, 3), padding='same', name='rb_conv_' + str(id) + '_2')(x)
    x = BatchNormalization(axis=axis_b, name="rb_batchnorm_" + str(id) + "_2")(x)

    m = Add()([x, init])
    return m

def sr_model_X2_1(img_width = None, img_height = None, axis_b = 1):
    scale = 2
    ip = Input(shape=(3, img_width, img_height), name="X_input")   
    # ip_ = Permute((3, 1, 2))(ip)
    ip_norm = Normalize(name='input_norm')(ip)

    conv1   = Conv2D(num_filters, (5, 5), padding='same', strides=(1,1), name='sr_conv1')(ip_norm)
    conv1_b = Activation('relu', name='sr_conv1_activ')(conv1)
    
    resblock = _residual_block(conv1_b,  1, axis_b)
    resblock = _residual_block(resblock, 2, axis_b)
    resblock = _residual_block(resblock, 3, axis_b)
    resblock = _residual_block(resblock, 4, axis_b)
    resblock = _residual_block(resblock, 5, axis_b)
    resblock = _residual_block(resblock, 6, axis_b)

    conv2 = Conv2D(num_filters, (3, 3), padding='same', strides=(1,1), name='sr_conv2')(resblock)
    conv2_b = BatchNormalization(axis=axis_b, name='sr_conv2_batchnorm')(conv2)
    conv2_b = Activation('relu', name='sr_conv2_activ')(conv2_b)

    merge1 = Add()([conv2_b, conv1_b])
    
    conv_up1 = Conv2D(num_filters * scale * scale, (3, 3), padding='same', strides=(1,1), name='sr_upconv1')(merge1)
    upsamp1 = SubPixelUpscaling(scale, num_filters)(conv_up1)
    upsamp1 = Activation('relu', name='sr_upconv1_activ')(upsamp1)

    tv_regularizer = TVRegularizer(img_width=im_size * scale, img_height=im_size * scale,
                                       weight=1e-4)

    conv3 = Conv2D(3, (5, 5), padding='same', strides=(1,1), name='sr_conv3', activation = 'tanh',
            activity_regularizer=tv_regularizer)(upsamp1)

    out = Denormalize()(conv3)
    model = Model(ip, out)
    # print model.summary()

    return model

def sr_model_X3_1(img_width = None, img_height = None, axis_b = 1):
    scale = 3
    ip = Input(shape=(3, img_width, img_height), name="X_input")   
    # ip_ = Permute((3, 1, 2))(ip)
    ip_norm = Normalize(name='input_norm')(ip)

    conv1   = Conv2D(num_filters, (5, 5), padding='same', strides=(1,1), name='sr_conv1')(ip_norm)
    conv1_b = Activation('relu', name='sr_conv1_activ')(conv1)
    
    resblock = _residual_block(conv1_b,  1, axis_b)
    resblock = _residual_block(resblock, 2, axis_b)
    resblock = _residual_block(resblock, 3, axis_b)
    resblock = _residual_block(resblock, 4, axis_b)
    resblock = _residual_block(resblock, 5, axis_b)
    resblock = _residual_block(resblock, 6, axis_b)
    resblock = _residual_block(resblock, 7, axis_b)
    resblock = _residual_block(resblock, 8, axis_b)

    conv2 = Conv2D(num_filters, (3, 3), padding='same', strides=(1,1), name='sr_conv2')(resblock)
    conv2_b = BatchNormalization(axis=axis_b, name='sr_conv2_batchnorm')(conv2)
    conv2_b = Activation('relu', name='sr_conv2_activ')(conv2_b)

    merge1 = Add()([conv2_b, conv1_b])
    
    conv_up1 = Conv2D(num_filters * scale * scale, (3, 3), padding='same', strides=(1,1), name='sr_upconv1')(merge1)
    upsamp1 = SubPixelUpscaling(scale, num_filters)(conv_up1)
    upsamp1 = Activation('relu', name='sr_upconv1_activ')(upsamp1)

    tv_regularizer = TVRegularizer(img_width=im_size * scale * scale, img_height=im_size * scale * scale,
                                       weight=1e-4)

    conv3 = Conv2D(3, (5, 5), padding='same', strides=(1,1), name='sr_conv3', activation = 'tanh',
                    activity_regularizer=tv_regularizer)(upsamp1)

    out = Denormalize()(conv3)
    model = Model(ip, out)
    # print model.summary()

    return model

def sr_model_X4_1(img_width = None, img_height = None, axis_b = 1):
    scale = 2
    ip = Input(shape=(3, img_width, img_height), name="X_input")   
    # ip_ = Permute((3, 1, 2))(ip)
    ip_norm = Normalize(name='input_norm')(ip)

    conv1   = Conv2D(num_filters, (5, 5), padding='same', strides=(1,1), name='sr_conv1')(ip_norm)
    conv1_b = Activation('relu', name='sr_conv1_activ')(conv1)
    
    resblock = _residual_block(conv1_b,  1, axis_b)
    resblock = _residual_block(resblock, 2, axis_b)
    resblock = _residual_block(resblock, 3, axis_b)
    resblock = _residual_block(resblock, 4, axis_b)
    resblock = _residual_block(resblock, 5, axis_b)
    resblock = _residual_block(resblock, 6, axis_b)
    resblock = _residual_block(resblock, 7, axis_b)
    resblock = _residual_block(resblock, 8, axis_b)
    resblock = _residual_block(resblock, 9, axis_b)
    resblock = _residual_block(resblock, 10, axis_b)

    conv2 = Conv2D(num_filters, (3, 3), padding='same', strides=(1,1), name='sr_conv2')(resblock)
    conv2_b = BatchNormalization(axis=axis_b, name='sr_conv2_batchnorm')(conv2)
    conv2_b = Activation('relu', name='sr_conv2_activ')(conv2_b)

    merge1 = Add()([conv2_b, conv1_b])
    
    conv_up1 = Conv2D(num_filters * scale * scale, (3, 3), padding='same', strides=(1,1), name='sr_upconv1')(merge1)
    upsamp1 = SubPixelUpscaling(scale, num_filters)(conv_up1)
    upsamp1 = Activation('relu', name='sr_upconv1_activ')(upsamp1)

    conv_up2 = Conv2D(num_filters * scale * scale, (3, 3), padding='same', strides=(1,1), name='sr_upconv2')(upsamp1)
    upsamp2 = SubPixelUpscaling(scale, num_filters)(conv_up2)
    upsamp2 = Activation('relu', name='sr_upconv2_activ')(upsamp2)

    tv_regularizer = TVRegularizer(img_width=im_size * scale * scale, img_height=im_size * scale * scale,
                                       weight=1e-4)

    conv3 = Conv2D(3, (5, 5), padding='same', strides=(1,1), name='sr_conv3', activation = 'tanh',
                   activity_regularizer=tv_regularizer)(upsamp2)

    out = Denormalize()(conv3)
    model = Model(ip, out)
    # print model.summary()

    return model

def sr_model2(img_width = None, img_height = None, axis_b = 1):
    ip = Input(shape=(3, img_width, img_height), name="X_input")   
    # ip_ = Permute((3, 1, 2))(ip)
    ip_norm = Normalize(name='input_norm')(ip)

    conv1   = Conv2D(num_filters, (5, 5), padding='same', strides=(1,1), name='sr_conv1')(ip_norm)
    conv1_b = Activation('relu', name='sr_conv1_activ')(conv1)
    
    resblock = _inception_residual_block(conv1_b,  1, axis_b)
    resblock = _inception_residual_block(resblock, 2, axis_b)
    resblock = _inception_residual_block(resblock, 3, axis_b)
    resblock = _inception_residual_block(resblock, 4, axis_b)
    resblock = _inception_residual_block(resblock, 5, axis_b)
    resblock = _inception_residual_block(resblock, 6, axis_b)
    resblock = _inception_residual_block(resblock, 7, axis_b)
    resblock = _inception_residual_block(resblock, 8, axis_b)

    conv2 = Conv2D(num_filters, (3, 3), padding='same', strides=(1,1), name='sr_conv2')(resblock)
    conv2_b = BatchNormalization(axis=axis_b, name='sr_conv2_batchnorm')(conv2)
    conv2_b = Activation('relu', name='sr_conv2_activ')(conv2_b)

    merge1 = Add()([conv2_b, conv1_b])
    
    conv_up1 = Conv2D(num_filters * scale * scale, (3, 3), padding='same', strides=(1,1), name='sr_upconv1')(merge1)
    upsamp1 = SubPixelUpscaling(scale, num_filters)(conv_up1)
    upsamp1 = Activation('relu', name='sr_upconv2_activ')(upsamp1)

    tv_regularizer = TVRegularizer(img_width=im_size * scale, img_height=im_size * scale,
                                       weight=1e-4)

    conv3 = Conv2D(3, (5, 5), padding='same', strides=(1,1), name='sr_conv3', activation = 'tanh')(upsamp1)

    out = Denormalize()(conv3)
    model = Model(ip, out)
    print model.summary()

    return model

def sr_model3(img_width = None, img_height = None, axis_b = 1):
    ip = Input(shape=(3, img_width, img_height), name="X_input")   
    # ip_ = Permute((3, 1, 2))(ip)
    ip_norm = Normalize(name='input_norm')(ip)

    conv1   = Conv2D(num_filters, (5, 5), padding='same', strides=(1,1), name='sr_conv1')(ip_norm)
    conv1_b = Activation('relu', name='sr_conv1_activ')(conv1)
    
    resblock = _residual_block(conv1_b,  1, axis_b)
    resblock = _residual_block(resblock, 2, axis_b)
    resblock = _residual_block(resblock, 3, axis_b)
    resblock = _residual_block(resblock, 4, axis_b)
    resblock = _residual_block(resblock, 5, axis_b)
    resblock = _residual_block(resblock, 6, axis_b)
    resblock = _residual_block(resblock, 7, axis_b)
    resblock = _residual_block(resblock, 8, axis_b)

    conv2 = Conv2D(num_filters, (3, 3), padding='same', strides=(1,1), name='sr_conv2')(resblock)
    conv2_b = BatchNormalization(axis=axis_b, name='sr_conv2_batchnorm')(conv2)
    conv2_b = Activation('relu', name='sr_conv2_activ')(conv2_b)

    merge1 = Add()([conv2_b, conv1_b])
    
    conv_up_x2 = Conv2D(num_filters * scale2 * scale2, (3, 3), padding='same', strides=(1,1), name='sr_upconv_x2')(merge1)
    upsamp_x2 = SubPixelUpscaling(scale2, num_filters)(conv_up_x2)
    upsamp_x2 = Activation('relu', name='sr_upconv2_activ')(upsamp_x2)

    conv_up_x3 = Conv2D(num_filters * scale3 * scale3, (3, 3), padding='same', strides=(1,1), name='sr_upconv_x3')(merge1)
    upsamp_x3 = SubPixelUpscaling(scale3, num_filters)(conv_up_x3)
    upsamp_x3 = Activation('relu', name='sr_upconv_x3_activ')(upsamp_x3)

    conv_up_x4 = Conv2D(num_filters * scale2 * scale2, (3, 3), padding='same', strides=(1,1), name='sr_upconv_x4')(upsamp_x2)
    upsamp_x4 = SubPixelUpscaling(scale2, num_filters)(conv_up_x4)
    upsamp_x4 = Activation('relu', name='sr_upconv_x4_activ')(upsamp_x4)

    # tv_regularizer = TVRegularizer(img_width=im_size * scale, img_height=im_size * scale,
    #                                    weight=1e-4)

    conv3_x2 = Conv2D(3, (5, 5), padding='same', strides=(1,1), name='sr_conv_x2', activation = 'tanh')(upsamp_x2)
    conv3_x3 = Conv2D(3, (5, 5), padding='same', strides=(1,1), name='sr_conv_x3', activation = 'tanh')(upsamp_x3)
    conv3_x4 = Conv2D(3, (5, 5), padding='same', strides=(1,1), name='sr_conv_x4', activation = 'tanh')(upsamp_x4)

    out_x2 = Denormalize()(conv3_x2)
    out_x3 = Denormalize()(conv3_x3)
    out_x4 = Denormalize()(conv3_x4)

    model = Model(ip, outputs=[out_x2, out_x3, out_x4])
    print model.summary()

    return model

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