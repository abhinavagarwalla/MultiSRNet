import os
import h5py
import time
import keras
import random
import itertools
import scipy.misc
from loss import *
import numpy as np
# from keras_model import sr_model1
from keras import backend as K
from sklearn.utils import shuffle
from imgaug import augmenters as iaa
from keras.engine.topology import Layer
from keras.utils.data_utils import get_file
from keras.models import Sequential, Model, load_model
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Dense, Dropout, Flatten, merge, Add, Input, Concatenate, Average
from keras.layers import Conv2D, MaxPooling2D, Permute, Activation, BatchNormalization

keras.backend.set_image_dim_ordering('th')

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

class Scale(Layer):
    '''
    Custom layer to scale the inputs by multiplying with a factor.
    '''

    def __init__(self, **kwargs):
        super(Scale, self).__init__(**kwargs)
        self.factor = 0.1

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        '''
        Since the input image has range [0, 255]
        '''

        return x * self.factor

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

def _residual_block(ip, id, axis_b):
    init = ip

    x = Conv2D(num_filters, (3, 3), padding='same', name='rb_conv_' + str(id) + '_1')(ip)
    x = BatchNormalization(axis=axis_b, name="rb_batchnorm_" + str(id) + "_1")(x)
    x = Activation('relu', name="rb_activation_" + str(id) + "_1")(x)

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

def get_augmentation(hr, lr):
    hflipper = iaa.Fliplr(1.0) # always horizontally flip each input image
    vflipper = iaa.Flipud(1.0) # vertically flip each input image with 90% probability

    hr = hr.astype(np.uint8)
    lr = lr.astype(np.uint8)

    hg, lg = [], []
    for i in range(len(hr)):
        x = random.random()
        if x < 1:
            hg.append(hflipper.augment_image(hr[i]))
            lg.append(hflipper.augment_image(lr[i]))
        elif x < 0.5:
            hg.append(vflipper.augment_image(hr[i]))
            lg.append(vflipper.augment_image(lr[i]))
    hr = np.append(hr, hg, axis=0)
    lr = np.append(lr, lg, axis=0)
    return hr.astype(np.float32), lr.astype(np.float32)

def make_submission(load_path, save_path):
    start_time =  time.clock()
    num_imgs = 0
    for filename in os.listdir(load_path):
        num_imgs += 1
        test_im = [scipy.misc.imread(load_path + filename)]
        # print test_im[0].shape
        sample = gen.predict([np.transpose(test_im, (0, 3, 1, 2))])
        sample = en_net.predict(sample)

        # print np.transpose(sample[0], (1, 2, 0)).shape
        scipy.misc.imsave(save_path + filename, np.transpose(sample[0], (1, 2, 0)))
    print ('time_per_image', (time.clock() - start_time) / num_imgs)

if __name__=='__main__':
    img_height = None
    img_width = None
    pool_type = 0

    hr_path = '../SuperResolution/patches/train_HR/X3/'
    lr_path = '../SuperResolution/patches/train_LR_unknown/X3/'

    save_path = '../SuperResolution/submission/valid_LR_unknown/X3/'
    load_path = '../SuperResolution/data/DIV2K_valid_LR_unknown/X3/'

    gen_model_name = 'model_resnet_aug_x3.h5'
    model_name = 'en_model_resnet_aug_x3.h5'
    load_checkpt = False
    train = True

    epochs = 50
    batch_size = 16
    im_size = 96
    scale = 3 
    max_score = 0.0
    num_filters = 64
    iterations = 1
    # sr_model1 -> resnet
    # sr_model2 -> inception-resnet
    gen = sr_model1()
    gen.load_weights('weights/' + gen_model_name)
    en_net = enhancement_model()

    if load_checkpt == True:
        en_net.load_weights('weights/' + model_name)
        en_net.compile(optimizer=keras.optimizers.Adam(lr=1e-4), loss='mse', metrics=[PSNRLoss])
    else:
        en_net.compile(optimizer=keras.optimizers.Adam(lr=1e-4), loss='mse', metrics=[PSNRLoss])

    if train == True:
        max_g_loss = 0.
        for epoch in xrange(epochs):
            idx = 0
            iscore = 0.0, 0.0 # self.get_inception_score()
            for filename in os.listdir(hr_path):
                hr_np = np.load(hr_path + filename)
                lr_np = np.load(lr_path + filename)

                # hr_np, lr_np = shuffle(hr_np, lr_np)
                # hr_np, lr_np = get_augmentation(hr_np, lr_np)

                for i in range(0, int(len(hr_np) / batch_size)):
                    lr_batch = lr_np[batch_size * i: batch_size * (i + 1)]
                    hr_batch = hr_np[batch_size * i: batch_size * (i + 1)]

                    sr_out = gen.predict(lr_batch.transpose((0, 3, 1, 2)))
                    g_loss = en_net.train_on_batch(sr_out, hr_batch.transpose((0, 3, 1, 2)))
                    print ("Epoch : ", epoch, " | Loss : ", g_loss) 
                    
                    iterations += 1

                    if iterations % 100 == 0:
                        iterations = 0
                        if g_loss[1] > max_g_loss:
                            max_g_loss = g_loss[1]
                            en_net.save_weights('weights/' + model_name[:-3] + '_best_on_train.h5')    
                            # make_submission(load_path, save_path)

            # if (epoch + 1) % 5 == 4:
            #     bsd100_lr = np.load('')
            #     bsd100_hr = np.load('')
            #     scores = gen.evaluate(bsd100_lr, bsd100_hr, batch_size=16)
            #     print scores

            #     if scores[1] > max_score:
            #         gen.save('weights/' + model_name)
            #         make_submission(load_path, save_path)
            #         max_score = scores[1]
