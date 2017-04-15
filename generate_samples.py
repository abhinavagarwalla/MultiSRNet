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

class VGGNormalize(Layer):
    '''
    Custom layer to subtract the outputs of previous layer by 120,
    to normalize the inputs to the VGG network.
    '''

    def __init__(self, **kwargs):
        super(VGGNormalize, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        '''
        Since individual channels cannot be altered in a TensorVariable, therefore
        we subtract it by 120, similar to the chainer implementation.
        '''
        if K.backend() == "theano":
            import theano.tensor as T
            #x = T.set_subtensor(x[:, :, :, :], x[:, ::-1, :, :]) # RGB -> BGR
            x = T.set_subtensor(x[:, 0, :, :], x[:, 0, :, :] - 103.939)
            x = T.set_subtensor(x[:, 1, :, :], x[:, 1, :, :] - 116.779)
            x = T.set_subtensor(x[:, 2, :, :], x[:, 2, :, :] - 123.680)
            #x -= 120
        else:
            # No exact substitute for set_subtensor in tensorflow
            # So we subtract an approximate value
            x -= 120
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

def pooling_func(x, pooltype):
    '''
    Pooling function used in VGG
    Args:
        x: previous layer
        pooltype: int, 1 refers to AveragePooling2D. All other values refer to MaxPooling2D
    Returns:
    '''
    if pooltype == 1:
        return AveragePooling2D((2, 2), strides=(2, 2))(x)
    else:
        return MaxPooling2D((2, 2), strides=(2, 2))(x)

class VGG:
    '''
    Helper class to load VGG and its weights to the FastNet model
    '''

    def __init__(self, img_height=96, img_width=96):
        self.img_height = img_height
        self.img_width = img_width

    def append_vgg_model(self, model_input, x_in, pool_type=0):
        '''
        Adds the VGG model to the FastNet model. It concatenates the original input to the output generated
        by the FastNet model. This is used to compute output features of VGG for the input image.
        Then it rescales the FastNet outputs and initial input to range [-127.5, 127.5] with lambda layer
        Ideally, I would like to subtract the channel means individually, but that is not efficient.
        Therefore, the closest approximate is to scale the values in the range [-127.5, 127.5]
        After this it adds the VGG layers.
        Args:
            model_input: Input to the FastNet model
            x_in: Output of last layer of FastNet model
            pool_type: int, 1 = AveragePooling, otherwise uses MaxPooling
        Returns: Model (FastNet + VGG)
        '''
        true_X_input = Input(shape=(3, self.img_width, self.img_height))

        # Append the initial input to the FastNet input to the VGG inputs
        x = merge([x_in, true_X_input], mode='concat', concat_axis=0)

        # Normalize the inputs via custom VGG Normalization layer
        x = VGGNormalize(name="vgg_normalize")(x)

        # Begin adding the VGG layers
        x = Conv2D(64, (3, 3), activation='relu', name='conv1_1')(x)
        x = Conv2D(64, (3, 3), activation='relu', name='conv1_2', padding='same')(x)
        x = pooling_func(x, pool_type)

        x = Conv2D(128, (3, 3), activation='relu', name='conv2_1', padding='same')(x)
        x = Conv2D(128, (3, 3), activation='relu', name='conv2_2', padding='same')(x)
        x = pooling_func(x, pool_type)

        x = Conv2D(256, (3, 3), activation='relu', name='conv3_1', padding='same')(x)
        x = Conv2D(256, (3, 3), activation='relu', name='conv3_2', padding='same')(x)
        x = Conv2D(256, (3, 3), activation='relu', name='conv3_3', padding='same')(x)
        x = pooling_func(x, pool_type)

        x = Conv2D(512, (3, 3), activation='relu', name='conv4_1', padding='same')(x)
        x = Conv2D(512, (3, 3), activation='relu', name='conv4_2', padding='same')(x)
        x = Conv2D(512, (3, 3), activation='relu', name='conv4_3', padding='same')(x)
        x = pooling_func(x, pool_type)

        x = Conv2D(512, (3, 3), activation='relu', name='conv5_1', padding='same')(x)
        x = Conv2D(512, (3, 3), activation='relu', name='conv5_2', padding='same')(x)
        x = Conv2D(512, (3, 3), activation='relu', name='conv5_3', padding='same')(x)
        x = pooling_func(x, pool_type)

        model = Model([model_input, true_X_input], x)

        # Loading VGG 16 weights
        if K.image_dim_ordering() == "th":
            weights_name = "vgg16_weights_th_dim_ordering_th_kernels_notop.h5"
            weights_path = ""#THEANO_WEIGHTS_PATH_NO_TOP
        else:
            weights_name = "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
            weights_path = TENSORFLOW_WEIGHTS_PATH_NO_TOP

        # weights = get_file(weights_name, weights_path, cache_subdir='.')
        f = h5py.File(weights_name)

        layer_names = [name for name in f.attrs['layer_names']]
        print layer_names

        for i, layer in enumerate(model.layers[-18:]):
            g = f[layer_names[i]]
            weights = [np.asarray(g[name]).transpose() for name in g.attrs['weight_names']]
            layer.set_weights(weights)
        print('VGG Model weights loaded.')

        # Freeze all VGG layers
        for layer in model.layers[-19:]:
            layer.trainable = False

        return model

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

def sr_model1(img_width = None, img_height = None, axis_b = 1):
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
    upsamp1 = Activation('relu', name='sr_upconv2_activ')(upsamp1)

    tv_regularizer = TVRegularizer(img_width=im_size * scale, img_height=im_size * scale,
                                       weight=1e-4)

    conv3 = Conv2D(3, (5, 5), padding='same', strides=(1,1), name='sr_conv3', activation = 'tanh')(upsamp1)

    out = Denormalize()(conv3)
    model = Model(ip, out)
    print model.summary()

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

def make_submission(load_path, save_path):
    start_time =  time.clock()
    num_imgs = 0
    for filename in os.listdir(load_path):
        num_imgs += 1
        test_im = [scipy.misc.imread(load_path + filename)]
        # print test_im[0].shape
        sample = gen.predict([np.transpose(test_im, (0, 3, 1, 2))])
        # print np.transpose(sample[0], (1, 2, 0)).shape
        scipy.misc.imsave(save_path + filename, np.transpose(sample[0], (1, 2, 0)))
    print ('time_per_image', (time.clock() - start_time) / num_imgs)

def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def cross_validate():
    num_imgs = 0
    score = 0.0
    for filename in os.listdir(valid_lr_path):
        num_imgs += 1
        lr_inp = np.transpose([scipy.misc.imread(valid_lr_path + filename)], (0, 3, 1, 2))
        hr_inp = np.transpose([scipy.misc.imread(valid_hr_path + filename.replace('x' + str(scale), ''))], (0, 3, 1, 2))

        score += psnr(hr_inp, gen.predict(lr_inp))        
    return (score / num_imgs)

def cross_validate_en():
    num_imgs = 0
    score = 0.0
    for filename in os.listdir(valid_lr_path):
        num_imgs += 1
        lr_inp = np.transpose([scipy.misc.imread(valid_lr_path + filename)], (0, 3, 1, 2))
        hr_inp = np.transpose([scipy.misc.imread(valid_hr_path + filename.replace('x' + str(scale), ''))], (0, 3, 1, 2))

        score += psnr(hr_inp, en_net.predict(gen.predict(lr_inp)))   
        print score/num_imgs     
    return (score / num_imgs)

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
    # conv3 = Scale()(conv3)

    out = Average()([ip_norm, conv5])
    out = Denormalize()(out)

    en_net = Model(ip, out)
    print en_net.summary()
    return en_net

if __name__=='__main__':
    img_height = None
    img_width = None
    pool_type = 0

    user = '/home/abhinav/'
    data = 'DIV2K'
    scale = 2

    hr_path = '../SuperResolution/patches/DIV2K_train_HR/X' + str(scale) + '/'
    lr_path = '../SuperResolution/patches/DIV2K_train_LR_unknown/X' + str(scale) + '/'

    save_path = '../SuperResolution/submission/DIV2K_test_LR_unknown/X' + str(scale) + '/'
    load_path = '../SuperResolution/data/DIV2K_test_LR_unknown/X' + str(scale) + '/'

    valid_lr_path = '../SuperResolution/data/DIV2K_valid_LR_unknown/X' + str(scale) + '/'
    valid_hr_path = '../SuperResolution/data/DIV2K_valid_HR/'

    model_name = 'model_resnet_aug.h5'

    epochs = 50
    batch_size = 32
    im_size = 96
    max_score = 0.0
    num_filters = 64

    gen = sr_model1()
    gen.load_weights('weights/from_top/' + model_name)
    gen.compile(optimizer=keras.optimizers.Adam(lr=1e-4), loss='mse', metrics=[PSNRLoss])

    en_net = enhancement_model()
    en_net.load_weights('weights/en_model2.h5')

    print cross_validate_en()
    # scores = 0
    # for vi in range(1):
    #     bsd100_hr = np.load(user+'SuperResolution/patches/BSD_train_HR/X2/data_'+str(vi)+'.npy').transpose((0, 3, 1, 2))
    #     bsd100_lr = np.load(user+'SuperResolution/patches/BSD_train_LR_unknown/X2/data_'+str(vi)+'.npy').transpose((0, 3, 1, 2))
    #     scores += gen.evaluate(bsd100_lr, bsd100_hr, batch_size=16)[1]
    # scores = np.mean(scores)
    # print "Validating Data PSNR: ", scores

    # make_submission(load_path, save_path)