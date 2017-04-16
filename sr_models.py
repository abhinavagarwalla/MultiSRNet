from keras import backend as K
from keras.models import Model
from keras.layers import Input, merge, BatchNormalization, LeakyReLU, Add, Flatten, Dense, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.utils.data_utils import get_file

from keras_ops import smooth_gan_labels

from layers import *
from loss import *

import os
import time
import h5py
import keras
import numpy as np
import json
from keras.engine.topology import Layer
from scipy.misc import imresize, imsave
from scipy.ndimage.filters import gaussian_filter

keras.backend.set_image_dim_ordering('th')

THEANO_WEIGHTS_PATH_NO_TOP = r'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_th_dim_ordering_th_kernels_notop.h5'
TF_WEIGHTS_PATH_NO_TOP = r"https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"

if not os.path.exists("weights/"):
    os.makedirs("weights/")

if not os.path.exists("val_images/"):
    os.makedirs("val_images/")

if K.image_dim_ordering() == "th":
    channel_axis = 1
else:
    channel_axis = -1

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

class VGGNetwork:
    '''
    Helper class to load VGG and its weights to the FastNet model
    '''

    def __init__(self, img_width=384, img_height=384, vgg_weight=1e-3):
        self.img_height = img_height
        self.img_width = img_width
        self.vgg_weight = vgg_weight

        self.vgg_layers = None

    def append_vgg_network(self, x_in, true_X_input, pre_train=False):

        # Append the initial inputs to the outputs of the SRResNet
        x = merge([x_in, true_X_input], mode='concat', concat_axis=0)

        # Normalize the inputs via custom VGG Normalization layer
        x = VGGNormalize(name="normalize_vgg")(x)

        # Begin adding the VGG layers
        x = Conv2D(64, (3, 3), activation='relu', name='vgg_conv1_1', padding='same')(x)

        x = Conv2D(64, (3, 3), activation='relu', name='vgg_conv1_2', padding='same')(x)
        x = MaxPooling2D(name='vgg_maxpool1')(x)

        x = Conv2D(128, (3, 3), activation='relu', name='vgg_conv2_1', padding='same')(x)

        # if not pre_train:
        #     raw_input('in this')
        vgg_regularizer2 = ContentVGGRegularizer(weight=self.vgg_weight)
        x = Conv2D(128, (3, 3), activation='relu', name='vgg_conv2_2', padding='same',
                          activity_regularizer=vgg_regularizer2)(x)
        # else:
        #     x = Conv2D(128, (3, 3), activation='relu', name='vgg_conv2_2', padding='same')(x)
        x = MaxPooling2D(name='vgg_maxpool2')(x)

        x = Conv2D(256, (3, 3), activation='relu', name='vgg_conv3_1', padding='same')(x)
        x = Conv2D(256, (3, 3), activation='relu', name='vgg_conv3_2', padding='same')(x)

        x = Conv2D(256, (3, 3), activation='relu', name='vgg_conv3_3', padding='same')(x)
        x = MaxPooling2D(name='vgg_maxpool3')(x)

        x = Conv2D(512, (3, 3), activation='relu', name='vgg_conv4_1', padding='same')(x)
        x = Conv2D(512, (3, 3), activation='relu', name='vgg_conv4_2', padding='same')(x)

        x = Conv2D(512, (3, 3), activation='relu', name='vgg_conv4_3', padding='same')(x)
        x = MaxPooling2D(name='vgg_maxpool4')(x)

        x = Conv2D(512, (3, 3), activation='relu', name='vgg_conv5_1', padding='same')(x)
        x = Conv2D(512, (3, 3), activation='relu', name='vgg_conv5_2', padding='same')(x)

        # if not pre_train:
        # vgg_regularizer5 = ContentVGGRegularizer(weight=self.vgg_weight)
        # x = Conv2D(512, (3, 3), activation='relu', name='vgg_conv5_3', padding='same',
                      # activity_regularizer=vgg_regularizer5)(x)
        # else:
        x = Conv2D(512, (3, 3), activation='relu', name='vgg_conv5_3', padding='same')(x)
        x = MaxPooling2D(name='vgg_maxpool5')(x)

        return x

    def load_vgg_weight(self, model):
        # Loading VGG 16 weights
        # vgg16_weights_th_dim_ordering_th_kernels_notop.h5 
        # if K.image_dim_ordering() == "th":
        #     weights = get_file('vgg16_weights_th_dim_ordering_th_kernels_notop.h5', THEANO_WEIGHTS_PATH_NO_TOP,
        #                            cache_subdir='.')
        # else:
        #     weights = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', TF_WEIGHTS_PATH_NO_TOP,
        #                            cache_subdir='models')
        f = h5py.File('vgg16_weights_th_dim_ordering_th_kernels_notop.h5')

        layer_names = [name for name in f.attrs['layer_names']]

        if self.vgg_layers is None:
            self.vgg_layers = [layer for layer in model.layers
                               if 'vgg_' in layer.name]

        for i, layer in enumerate(self.vgg_layers):
            g = f[layer_names[i]]
            weights = [np.asarray(g[name]).transpose() for name in g.attrs['weight_names']]
            # weights = [g[name] for name in g.attrs['weight_names']]
            layer.set_weights(weights)

        # Freeze all VGG layers
        print 'here'
        for layer in self.vgg_layers:
            layer.trainable = False

        return model

class DiscriminatorNetwork:

    def __init__(self, img_width=384, img_height=384, adversarial_loss_weight=1, small_model=False):
        self.img_width = img_width
        self.img_height = img_height
        self.adversarial_loss_weight = adversarial_loss_weight
        self.small_model = small_model

        self.k = 3
        self.mode = 2
        self.weights_path = "weights/Discriminator weights.h5"

        self.gan_layers = None

    def append_gan_network(self, true_X_input):

        # Normalize the inputs via custom VGG Normalization layer
        x = Normalize()(true_X_input)

        x = Conv2D(64, (self.k, self.k), padding='same', name='gan_conv1_1')(x)
        x = LeakyReLU(0.3, name="gan_lrelu1_1")(x)

        x = Conv2D(64, (self.k, self.k), padding='same', name='gan_conv1_2', strides=(2, 2))(x)
        x = LeakyReLU(0.3, name='gan_lrelu1_2')(x)
        x = BatchNormalization(axis=channel_axis, name='gan_batchnorm1_1')(x)

        filters = [128, 256] #if self.small_model else [128, 256, 512]

        for i, nb_filters in enumerate(filters):
            for j in range(2):
                subsample = (2, 2) if j == 1 else (1, 1)

                x = Conv2D(nb_filters, (self.k, self.k), padding='same', strides=subsample,
                                  name='gan_conv%d_%d' % (i + 2, j + 1))(x)
                x = LeakyReLU(0.3, name='gan_lrelu_%d_%d' % (i + 2, j + 1))(x)
                x = BatchNormalization( axis=channel_axis, name='gan_batchnorm%d_%d' % (i + 2, j + 1))(x)

        x = Flatten(name='gan_flatten')(x)

        output_dim = 128 #if self.small_model else 1024

        x = Dense(output_dim, name='gan_dense1')(x)
        x = LeakyReLU(0.3, name='gan_lrelu5')(x)

        gan_regulrizer = AdversarialLossRegularizer(weight=self.adversarial_loss_weight)
        x = Dense(2, activation="softmax", activity_regularizer=gan_regulrizer, name='gan_output')(x)

        return x

    def set_trainable(self, model, value=True):
        if self.gan_layers is None:
            disc_model = [layer for layer in model.layers
                          if 'model' in layer.name][0] # Only disc model is an inner model

            self.gan_layers = [layer for layer in disc_model.layers
                               if 'gan_' in layer.name]

        for layer in self.gan_layers:
            layer.trainable = value

    def load_gan_weights(self, model):
        f = h5py.File(self.weights_path)

        layer_names = [name for name in f.attrs['layer_names']]
        layer_names = layer_names[1:] # First is an input layer. Not needed.

        if self.gan_layers is None:
            self.gan_layers = [layer for layer in model.layers
                                if 'gan_' in layer.name]

        for i, layer in enumerate(self.gan_layers):
            g = f[layer_names[i]]
            weights = [g[name] for name in g.attrs['weight_names']]
            layer.set_weights(weights)

        print("GAN Model weights loaded.")
        return model

    def save_gan_weights(self, model):
        print('GAN Weights are being saved.')
        model.save_weights(self.weights_path, overwrite=True)
        print('GAN Weights saved.')

class GenerativeNetwork:

    def __init__(self, img_width=96, img_height=96, batch_size=16, nb_upscales=1, scale = 2, small_model=False,
                 content_weight=1, tv_weight=1e-4, gen_channels=64):
        self.img_width = img_width
        self.img_height = img_height
        self.batch_size = batch_size
        self.small_model = small_model
        self.nb_scales = nb_upscales
        self.scale = scale

        self.content_weight = content_weight
        self.tv_weight = tv_weight

        self.filters = gen_channels
        self.init = 'glorot_uniform'

        self.sr_res_layers = None
        self.sr_weights_path = "weights/SRGAN.h5"

        self.output_func = None

    def create_sr_model(self, ip):

        x = Normalize()(ip)

        # ip_ = Permute((3, 1, 2))(ip)
        ip_norm = Normalize(name='input_norm')(ip)

        conv1   = Conv2D(64, (5, 5), padding='same', strides=(1,1), name='sr_conv1')(ip_norm)
        conv1_b = Activation('relu', name='sr_conv1_activ')(conv1)
        
        resblock = self._residual_block(conv1_b,  1)
        resblock = self._residual_block(resblock, 2)
        resblock = self._residual_block(resblock, 3)
        resblock = self._residual_block(resblock, 4)
        resblock = self._residual_block(resblock, 5)

        conv2 = Conv2D(64, (3, 3), padding='same', strides=(1,1), name='sr_conv2')(resblock)
        conv2_b = BatchNormalization(axis=channel_axis, name='se_conv2_batchnorm')(conv2)
        conv2_b = Activation('relu', name='sr_conv2_activ')(conv2_b)

        x = Add()([conv2_b, conv1_b])
        
        for i in range(self.nb_scales):
            x = self._upscale_block(x, i + 1)

        tv_regularizer = TVRegularizer(img_width=self.img_width * self.scale, img_height=self.img_height * self.scale,
                                       weight=self.tv_weight) #self.tv_weight)

        conv3 = Conv2D(3, (5, 5), padding='same', strides=(1,1), name='sr_conv3', activation = 'tanh',activity_regularizer=tv_regularizer)(x)

        out = Denormalize(name='sr_res_conv_denorm')(conv3)

        # x = Conv2D(self.filters, (3, 3), activation='linear', padding='same', name='sr_res_conv1',
        #                   kernel_initializer=self.init)(x)
        # # x = BatchNormalization(axis=channel_axis,  name='sr_res_bn_1')(x)
        # x = LeakyReLU(alpha=0.25, name='sr_res_lr1')(x)

        # # x = Conv2D(self.filters, 5, 5, activation='linear', padding='same', name='sr_res_conv2')(x)
        # # x = BatchNormalization(axis=channel_axis,  name='sr_res_bn_2')(x)
        # # x = LeakyReLU(alpha=0.25, name='sr_res_lr2')(x)

        # nb_residual = 5 if self.small_model else 15

        # for i in range(nb_residual):
        #     x = self._residual_block(x, i + 1)

        # for scale in range(self.nb_scales):
        #     x = self._upscale_block(x, scale + 1)

        # tv_regularizer = TVRegularizer(img_width=self.img_width * self.scale, img_height=self.img_height * self.scale,
        #                                weight=self.tv_weight) #self.tv_weight)

        # x = Conv2D(3, (3, 3), activation='tanh', padding='same', activity_regularizer=tv_regularizer,
        #                   kernel_initializer=self.init, name='sr_res_conv_final')(x)

        # x = Denormalize(name='sr_res_conv_denorm')(x)

        return out

    def _residual_block(self, ip, id):
        init = ip

        x = Conv2D(self.filters, (3, 3), activation='linear', padding='same', name='sr_res_conv_' + str(id) + '_1',
                          kernel_initializer=self.init)(ip)
        x = BatchNormalization(axis=channel_axis,  name='sr_res_bn_' + str(id) + '_1')(x)
        x = LeakyReLU(alpha=0.25, name="sr_res_activation_" + str(id) + "_1")(x)

        x = Conv2D(self.filters, (3, 3), activation='linear', padding='same', name='sr_res_conv_' + str(id) + '_2',
                          kernel_initializer=self.init)(x)
        x = BatchNormalization(axis=channel_axis,  name='sr_res_bn_' + str(id) + '_2')(x)

        m = merge([x, init], mode='sum', name="sr_res_merge_" + str(id))

        return m

    def _upscale_block(self, ip, id):
        '''
        As per suggestion from http://distill.pub/2016/deconv-checkerboard/, I am swapping out
        SubPixelConvolution to simple Nearest Neighbour Upsampling
        '''
        init = ip

        x = Conv2D(self.filters * self.scale * self.scale, (3, 3), padding='same', strides=(1,1), name='sr_res_upconv_' + str(id))(ip)
        x = SubPixelUpscaling(self.scale, self.filters)(x)
        x = LeakyReLU(alpha=0.25, name="sr_res_upscale_activation_" + str(id) + "_1")(x)

        # x = Conv2D(128, 3, 3, activation="linear", padding='same', name='sr_res_upconv1_%d' % id,
        #                   init=self.init)(init)
        # x = LeakyReLU(alpha=0.25, name='sr_res_up_lr_%d_1_1' % id)(x)
        # x = UpSampling2D(name='sr_res_upscale_%d' % id)(x)
        # #x = SubPixelUpscaling(r=2, channels=32)(x)
        # x = Conv2D(128, 3, 3, activation="linear", padding='same', name='sr_res_filter1_%d' % id,
        #                   init=self.init)(x)
        # x = LeakyReLU(alpha=0.3, name='sr_res_up_lr_%d_1_2' % id)(x)

        return x

    def set_trainable(self, model, value=True):
        if self.sr_res_layers is None:
            self.sr_res_layers = [layer for layer in model.layers
                                    if 'sr_res_' in layer.name]

        for layer in self.sr_res_layers:
            layer.trainable = value

    def get_generator_output(self, input_img, srgan_model):
        if self.output_func is None:
            gen_output_layer = [layer for layer in srgan_model.layers
                                if layer.name == "sr_res_conv_denorm"][0]
            self.output_func = K.function([srgan_model.layers[0].input],
                                          [gen_output_layer.output])

        return self.output_func([input_img])

class SRGANNetwork:

    def __init__(self, img_width=96, img_height=96, batch_size=16, nb_scales=2, scale=2):
        self.img_width = img_width
        self.img_height = img_height
        self.batch_size = batch_size
        self.nb_scales = nb_scales
        self.scale = scale

        self.discriminative_network = None # type: DiscriminatorNetwork
        self.generative_network = None # type: GenerativeNetwork
        self.vgg_network = None # type: VGGNetwork

        self.srgan_model_ = None # type: Model
        self.generative_model_ = None # type: Model
        self.discriminative_model_ = None #type: Model

    def build_srgan_model(self, use_small_srgan=False, use_small_discriminator=False):
        large_width = self.img_width * self.scale
        large_height = self.img_height * self.scale

        self.generative_network = GenerativeNetwork(self.img_width, self.img_height, self.batch_size, nb_upscales=self.nb_scales,
                                                    small_model=use_small_srgan)
        self.discriminative_network = DiscriminatorNetwork(large_width, large_height,
                                                           small_model=use_small_discriminator)
        self.vgg_network = VGGNetwork(large_width, large_height)

        # ip_test = Input(shape=(3, None, None), name='x_generator')
        ip= Input(shape=(3, self.img_height, self.img_width), name='x_gen_gan')
        ip_gan = Input(shape=(3, large_width, large_height), name='x_discriminator') # Actual X images
        ip_vgg = Input(shape=(3, large_height, large_width), name='x_vgg') # Actual X images

        sr_output = self.generative_network.create_sr_model(ip)
        self.generative_model_ = Model(ip, sr_output)
        # print self.generative_model_.summary()    

        gan_output = self.discriminative_network.append_gan_network(ip_gan)
        self.discriminative_model_ = Model(ip_gan, gan_output)

        gan_output = self.discriminative_model_(self.generative_model_.output)
        vgg_output = self.vgg_network.append_vgg_network(self.generative_model_.output, ip_vgg)

        self.srgan_model_ = Model(input=ip, output=gan_output)
        # print self.srgan_model_.summary()

        self.vgg_network.load_vgg_weight(self.srgan_model_)

        # self.test_network = Model(ip_test, self.generative_model_(ip_test))
        # self.test_network.summary()

    def train_full_model(self, hr_path, lr_path, nb_epochs=10, use_small_srgan=False,
                         use_small_discriminator=False):

        self.build_srgan_model(use_small_srgan, use_small_discriminator)

        self._train_model(hr_path, lr_path, batch_size, nb_epochs, load_generative_weights=True, load_discriminator_weights=True)

    def _train_model(self, hr_path, lr_path, nb_epochs=10, pre_train_srgan=False,
                     pre_train_discriminator=False, load_generative_weights=False, load_discriminator_weights=False,
                     save_loss=True, disc_train_flip=0.1):

        assert self.img_width >= 16, "Minimum image width must be at least 16"
        assert self.img_height >= 16, "Minimum image height must be at least 16"

        if load_generative_weights:
            try:
                self.generative_model_.load_weights(self.generative_network.sr_weights_path)
                print("Generator weights loaded.")
            except:
                print("Could not load generator weights.")

        if load_discriminator_weights:
            try:
                self.discriminative_network.load_gan_weights(self.srgan_model_)
                print("Discriminator weights loaded.")
            except:
                print("Could not load discriminator weights.")

        datagen = ImageDataGenerator(rescale=1. / 255)
        img_width = self.img_width * self.scale
        img_height = self.img_height * self.scale

        early_stop = False
        iteration = 0
        prev_improvement = -1

        if save_loss:
            if pre_train_srgan:
                loss_history = {'generator_loss' : [],
                                'val_psnr' : [], }
            elif pre_train_discriminator:
                loss_history = {'discriminator_loss' : [],
                                'discriminator_acc' : [], }
            else:
                loss_history = {'discriminator_loss' : [],
                                'discriminator_acc' : [],
                                'generator_loss' : [],
                                'val_psnr': [], }

        y_vgg_dummy = np.zeros((self.batch_size, 3, img_width // 32, img_height // 32)) # 5 Max Pools = 2 ** 5 = 32
        
        srgan_optimizer = Adam(lr=1e-4)
        generator_optimizer = Adam(lr=1e-4)
        discriminator_optimizer = Adam(lr=1e-4)

        self.generative_model_.compile(generator_optimizer, loss='mse')
        self.discriminative_model_.compile(discriminator_optimizer, loss='categorical_crossentropy', metrics=['acc'])
        self.srgan_model_.compile(srgan_optimizer, loss='categorical_crossentropy', metrics=['acc'])

        valid_filenames = []
        print("Training SRGAN network")
        for epoch in xrange(nb_epochs):
            idx = 0
            iscore = 0.0, 0.0 # self.get_inception_score()
            validation_psnr = 0.0
            validation_num= 0.0

            for filename in os.listdir(hr_path):
                hr_np = np.load(hr_path + filename)
                lr_np = np.load(lr_path + filename)

                for i in range(0, int(len(hr_np) / self.batch_size)):
                    batchX = lr_np[self.batch_size * i: self.batch_size * (i + 1)]
                    batchY = hr_np[self.batch_size * i: self.batch_size * (i + 1)]

                    batchX = batchX.transpose((0, 3, 1, 2))
                    batchY = batchY.transpose((0, 3, 1, 2))

                    if filename in valid_filenames:
                        # print("Validation image..")
                        # output_image_batch = self.generative_model_.predict(batchX)
                        # if type(output_image_batch) == list:
                        #     output_image_batch = output_image_batch[0]

                        # mean_axis = (0, 2, 3) if K.image_dim_ordering() == 'th' else (0, 1, 2)

                        # average_psnr = 0.0

                        # print('gen img mean :', np.mean(output_image_batch / 255., axis=mean_axis))
                        # print('val img mean :', np.mean(batchY / 255., axis=mean_axis))

                        # for x_i in range(self.batch_size):
                        #     average_psnr += psnr(x[x_i], np.clip(output_image_batch[x_i], 0, 255) / 255.)

                        # validation_num += self.batch_size
                        # validation_psnr += average_psnr
                        # if save_loss:
                        #     loss_history['val_psnr'].append(average_psnr)

                        # print("Time required : %0.2f. Average validation PSNR over %d samples = %0.2f" %
                        #       (t2 - t1, self.batch_size, average_psnr))

                        # for x_i in range(self.batch_size):
                        #     real_path = "val_images/epoch_%d_iteration_%d_num_%d_real_.png" % (i + 1, iteration, x_i + 1)
                        #     generated_path = "val_images/epoch_%d_iteration_%d_num_%d_generated.png" % (i + 1,
                        #                                                                                 iteration,
                        #                                                                                 x_i + 1)

                        #     val_x = x[x_i].copy() * 255.
                        #     val_x = val_x.transpose((1, 2, 0))
                        #     val_x = np.clip(val_x, 0, 255).astype('uint8')

                        #     output_image = output_image_batch[x_i]
                        #     output_image = output_image.transpose((1, 2, 0))
                        #     output_image = np.clip(output_image, 0, 255).astype('uint8')

                        #     imsave(real_path, val_x)
                        #     imsave(generated_path, output_image)

                        # '''
                        # Don't train of validation images for now.
                        # Note that if nb_epochs > 1, there is a chance that
                        # validation images may be used for training purposes as well.
                        # In that case, this isn't strictly a validation measure, instead of
                        # just a check to see what the network has learned.
                        # '''
                        continue

                    elif pre_train_srgan:
                        # Train only generator + vgg network

                        # Use custom bypass_fit to bypass the check for same input and output batch size
                        # hist = bypass_fit(self.srgan_model_, [x_generator, x * 255], y_vgg_dummy,
                                                     # batch_size=self.batch_size, nb_epoch=1, verbose=0)
                        
                        sr_loss = self.generative_model_.train_on_batch(batchX, batchY) #hist.history['loss'][0]

                        # if save_loss:
                        #     loss_history['generator_loss'].extend(hist.history['loss'])

                        # if prev_improvement == -1:
                        #     prev_improvement = sr_loss

                        # improvement = (prev_improvement - sr_loss) / prev_improvement * 100
                        # prev_improvement = sr_loss

                        # iteration += self.batch_size
                        # t2 = time.time()

                        print("Epoch : %d | Generative Loss : %0.2f" % (epoch, sr_loss))
                    elif pre_train_discriminator:
                        # Train only discriminator
                        X_pred = self.generative_model_.predict(batchX)

                        X = np.concatenate((X_pred, batchY))

                        # Using soft and noisy labels
                        if np.random.uniform() > disc_train_flip:
                            # give correct classifications
                            y_gan = [0] * self.batch_size + [1] * self.batch_size
                        else:
                            # give wrong classifications (noisy labels)
                            y_gan = [1] * self.batch_size + [0] * self.batch_size

                        y_gan = np.asarray(y_gan, dtype=np.int).reshape(-1, 1)
                        y_gan = to_categorical(y_gan, num_classes=2)
                        y_gan = smooth_gan_labels(y_gan)

                        disc_loss = self.discriminative_model_.train_on_batch(X, y_gan)
                        # hist = self.discriminative_model_.fit(X, y_gan, batch_size=self.batch_size,
                        #                                       nb_epoch=1, verbose=0)

                        # discriminator_loss = hist.history['loss'][-1]
                        # discriminator_acc = hist.history['acc'][-1]

                        # if save_loss:
                        #     loss_history['discriminator_loss'].extend(hist.history['loss'])
                        #     loss_history['discriminator_acc'].extend(hist.history['acc'])

                        # if prev_improvement == -1:
                        #     prev_improvement = discriminator_loss

                        # improvement = (prev_improvement - discriminator_loss) / prev_improvement * 100
                        # prev_improvement = discriminator_loss

                        # iteration += self.batch_size
                        # t2 = time.time()

                        print("Epoch : %d | Discriminator Loss : %0.4f" % (epoch, disc_loss))

                    else:
                        # Train only discriminator, disable training of srgan
                        self.discriminative_network.set_trainable(self.srgan_model_, value=True)
                        self.generative_network.set_trainable(self.srgan_model_, value=False)

                        # Use custom bypass_fit to bypass the check for same input and output batch size
                        # hist = bypass_fit(self.srgan_model_, [x_generator, x * 255, x_vgg],
                        #                          [y_gan, y_vgg_dummy],
                        #                          batch_size=self.batch_size, nb_epoch=1, verbose=0)

                        X_pred = self.generative_model_.predict(batchX)
                        X = np.concatenate((X_pred, batchY))

                        # Using soft and noisy labels
                        if np.random.uniform() > disc_train_flip:
                            # give correct classifications
                            y_gan = [0] * self.batch_size + [1] * self.batch_size
                        else:
                            # give wrong classifications (noisy labels)
                            y_gan = [1] * self.batch_size + [0] * self.batch_size

                        y_gan = np.asarray(y_gan, dtype=np.int).reshape(-1, 1)
                        y_gan = to_categorical(y_gan, num_classes=2)
                        y_gan = smooth_gan_labels(y_gan)
                        hist1 = self.discriminative_model_.train_on_batch(X, y_gan)
                        # discriminator_loss = hist1.history['loss'][-1]

                        # Train only generator, disable training of discriminator
                        self.discriminative_network.set_trainable(self.srgan_model_, value=False)
                        self.generative_network.set_trainable(self.srgan_model_, value=True)

                        # Using soft labels
                        y_model = [1] * self.batch_size
                        y_model = np.asarray(y_model, dtype=np.int).reshape(-1, 1)
                        y_model = to_categorical(y_model, num_classes=2)
                        y_model = smooth_gan_labels(y_model)

                        # Use custom bypass_fit to bypass the check for same input and output batch size
                        # hist2 = bypass_fit(self.srgan_model_, [x_generator, x, x_vgg], [y_model, y_vgg_dummy],
                        #                    batch_size=self.batch_size, nb_epoch=1, verbose=0)

                        hist2 = self.srgan_model_.train_on_batch(batchX, y_model)

                        # if save_loss:
                        #     loss_history['discriminator_loss'].extend(hist1.history['loss'])
                        #     loss_history['discriminator_acc'].extend(hist1.history['acc'])
                        #     loss_history['generator_loss'].extend(hist2.history['loss'])

                        # if prev_improvement == -1:
                        #     prev_improvement = discriminator_loss

                        # improvement = (prev_improvement - discriminator_loss) / prev_improvement * 100
                        # prev_improvement = discriminator_loss

                        # iteration += self.batch_size
                        # t2 = time.time()
                        print("Epoch : %d | Iter : %d | Discriminator Loss : %0.3f | Generative Loss : %0.3f" %
                              (epoch, iteration, hist1[0], hist2[0]))

            print("Saving model weights.")
            # Save predictive (SR network) weights
            self._save_model_weights(pre_train_srgan, pre_train_discriminator)
            # self._save_loss_history(loss_history, pre_train_srgan, pre_train_discriminator, save_loss)

                    # if iteration >= nb_images:
                    #     break

                # except KeyboardInterrupt:
                #     print("Keyboard interrupt detected. Stopping early.")
                #     early_stop = True
                #     break
            # try:
            #     print 'validation accuracy is', validation_psnr / validation_num
            # except:
            #     print 'Error is cross validation'
            #     pass
            iteration = 0
            if epoch % 5 == 3:
                make_submission(self.generative_model_, load_path, save_path)
            if early_stop:
                break

        print("Finished training SRGAN network. Saving model weights.")
        # Save predictive (SR network) weights
        self._save_model_weights(pre_train_srgan, pre_train_discriminator)
        self._save_loss_history(loss_history, pre_train_srgan, pre_train_discriminator, save_loss)

    def _save_model_weights(self, pre_train_srgan, pre_train_discriminator):
        if not pre_train_discriminator:
            self.generative_model_.save_weights(self.generative_network.sr_weights_path, overwrite=True)

        if not pre_train_srgan:
            # Save GAN (discriminative network) weights
            self.discriminative_network.save_gan_weights(self.discriminative_model_)

    def _save_loss_history(self, loss_history, pre_train_srgan, pre_train_discriminator, save_loss):
        if save_loss:
            print("Saving loss history")

            if pre_train_srgan:
                with open('pretrain losses - srgan.json', 'w') as f:
                    json.dump(loss_history, f)
            elif pre_train_discriminator:
                with open('pretrain losses - discriminator.json', 'w') as f:
                    json.dump(loss_history, f)
            else:
                with open('fulltrain losses.json', 'w') as f:
                    json.dump(loss_history, f)

            print("Saved loss history")

def make_submission(gen, load_path, save_path):
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

if __name__ == "__main__":

    hr_path = '/users/TeamVideoSummarization/SuperResolution/patches/train_HR/X2/'
    lr_path = '/users/TeamVideoSummarization/SuperResolution/patches/train_LR_bicubic/X2/'

    save_path = '/users/TeamVideoSummarization/SuperResolution/submission/valid_LR_bicubic/X2/'
    load_path = '/users/TeamVideoSummarization/SuperResolution/data/DIV2K_valid_LR_bicubic/X2/'


    # from keras.utils.visualize_util import plot

    '''
    Base Network manager for the SRGAN model
    Width / Height = 32 to reduce the memory requirement for the discriminator.
    Batch size = 1 is slower, but uses the least amount of gpu memory, and also acts as
    Instance Normalization (batch norm with 1 input image) which speeds up training slightly.
    '''

    srgan_network = SRGANNetwork(img_width=96, img_height=96, batch_size=32, nb_scales=1, scale=2)
    srgan_network.build_srgan_model()
    #plot(srgan_network.srgan_model_, 'SRGAN.png', show_shapes=True)

    # Pretrain the SRGAN network
    srgan_network._train_model(hr_path, lr_path, nb_epochs=5, pre_train_srgan=True, load_generative_weights=False, load_discriminator_weights=False)

    # Pretrain the discriminator network
    srgan_network._train_model(hr_path, lr_path, nb_epochs=2, load_generative_weights=True, load_discriminator_weights=False)

    # Fully train the SRGAN with VGG loss and Discriminator loss
    srgan_network._train_model(hr_path, lr_path, nb_epochs=25, load_generative_weights=True, load_discriminator_weights=True)
