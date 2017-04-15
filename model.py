from __future__ import division
import os
import time
import math
import itertools
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from ops import *
from utils import *
from vgg import *

F = tf.app.flags.FLAGS

def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))

class DCGAN(object):
    def __init__(self, sess, input_height=108, input_width=108, is_crop=True,
         batch_size=64, sample_num = 64, output_height=64, output_width=64,
         y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
         gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
         input_fname_pattern='*.jpg', checkpoint_dir=None, sample_dir=None):
        """

        Args:
          sess: TensorFlow session
          batch_size: The size of batch. Should be specified before training.
          y_dim: (optional) Dimension of dim for y. [None]
          z_dim: (optional) Dimension of dim for Z. [100]
          gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
          df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
          gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
          dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
          c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        """
        self.sess = sess

        self.batch_size = None #F.batch_size
        self.sample_num = sample_num

        self.input_height = None #96
        self.input_width = None #96
        self.upscale_factor = 2
        self.c_dim = c_dim

        self.dataset_name = dataset_name
        self.input_fname_pattern = input_fname_pattern
        self.checkpoint_dir = checkpoint_dir

        self.hr_path = '/users/TeamVideoSummarization/SuperResolution/patches/train_HR/'
        self.lr_path = '/users/TeamVideoSummarization/SuperResolution/patches/train_LR_bicubic/X2/'

        # data_dict = loadWeightsData('./vgg16.npy')

        self.build_model_sr()

    def build_model_sr(self):
        lr_image_dims = [self.input_height, self.input_width, self.c_dim]
        # hr_image_dims = [self.input_height * self.upscale_factor, self.input_width * self.upscale_factor, self.c_dim]

        self.lr_image = tf.placeholder(
            tf.float32, [self.batch_size,] + lr_image_dims, name='real_images')
        self.hr_image = tf.placeholder(
           tf.float32, [self.batch_size,] + lr_image_dims, name='sample_inputs')
        self.split_size = tf.placeholder(tf.int32, [4], 'split_size')

        self.SR, self.SR_ = self.sr_model1(self.lr_image)
        self.SR_sum = image_summary("SR", self.SR)

        self.mse_loss = tf.reduce_mean(tf.squared_difference(self.SR, self.hr_image))

        # # conv features of HR image
        # self.hr_image_ = (self.hr_image + 1) * 127.5
        # vgg_h = custom_vgg16(self.hr_image_, data_dict=data_dict)
        # feature_h = [vgg_h.conv1_2, vgg_h.conv2_2, vgg_h.conv3_3, vgg_h.conv4_3, vgg_h.conv5_3]

        # # conv features of LR Image 
        # self.lr_image_ = (self.lr_image + 1) * 127.5
        # vgg_l = custom_vgg16(self.lr_image_, data_dict=data_dict)
        # feature_l = [vgg_l.conv1_2, vgg_l.conv2_2, vgg_l.conv3_3, vgg_l.conv4_3, vgg_l.conv5_3]

        # compute feature loss
        # self.loss_f = tf.zeros(batchsize, tf.float32)
        # for f, f_ in zip(vgg_h, vgg_l):
        #     self.loss_f += lambda_f * tf.reduce_mean(tf.sub(f, f_) ** 2, [1, 2, 3])

        self.sr_loss_actual = self.mse_loss # self.loss_f
        self.sr_mse_loss = scalar_summary("mse_loss", self.mse_loss)
        t_vars = tf.trainable_variables()
        self.sr_vars = [var for var in t_vars if 'sr_' in var.name]

        self.saver = tf.train.Saver()

    def train_sr(self, config):
        # noise_data = np.ones((16, 64, 64, 3))
        # self.sess.run([self.SR],
        #                     feed_dict={self.lr_image: noise_data}
        #         )

        """Train DCGAN"""
        global_step = tf.placeholder(tf.int32, [], name="global_step_epochs")

        # learning_rate_sr = tf.train.exponential_decay(F.learning_rate, global_step,
        #                                              decay_steps=F.decay_step,
        #                                              decay_rate=F.decay_rate, staircase=True)

        sr_opt = tf.train.AdamOptimizer(F.learning_rate, beta1=F.beta1)

        sr_grads = sr_opt.compute_gradients(self.mse_loss, var_list=self.sr_vars)
        sr_optim = sr_opt.apply_gradients(sr_grads)

        merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(F.log_dir, self.sess.graph)

        try:
          tf.global_variables_initializer().run()
        except:
          tf.initialize_all_variables().run()

        counter = 0
        start_time = time.time()

        self.ra, self.rb = -1, 1

        for epoch in xrange(F.epoch):
            idx = 0
            iscore = 0.0, 0.0 # self.get_inception_score()
            for filename in os.listdir(self.hr_path):
                hr_np = np.load(self.hr_path + filename)
                lr_np = np.load(self.lr_path + filename)

                for i in range(0, int(len(hr_np) / F.batch_size)):
                    lr_batch = lr_np[F.batch_size * i: F.batch_size * (i + 1)]
                    hr_batch = hr_np[F.batch_size * i: F.batch_size * (i + 1)]
                    
                    summary_str, _,  srlossf = self.sess.run(
                        [merged, sr_optim, self.sr_loss_actual],
                        feed_dict={self.lr_image: lr_batch,
                                   self.hr_image: hr_batch, 
                                   global_step: epoch})
                    self.writer.add_summary(summary_str, counter)
                    
                    errSR_actual = self.sr_loss_actual.eval({self.lr_image: lr_batch, 
                                                            self.hr_image: hr_batch})
                    # lrateSR = learning_rate_SR.eval({global_step: epoch})

                    counter += 1
                    idx += 1
                    print(("Epoch:[%2d] l_SR:%.2e sr_loss:%.8f")
                          % (epoch, F.learning_rate, errSR_actual))

                    # if np.mod(counter, 100) == 1:
                    #     samples, d_loss, g_loss = self.sess.run(
                    #         [self.G_mean, self.d_loss, self.g_loss_actual],
                    #         feed_dict={self.z: sample_z, self.images: zip(*sample_images)[0], self.text_emb: zip(*sample_images)[1],
                    #                     self.wrong_images: fake_img_reshuffle}
                    #     )
                    #     save_images(samples, [8, 8],
                    #                 F.sample_dir + "/sample.png")
                    #     print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))

                    if np.mod(counter, 50) == 2:
                        self.save(F.checkpoint_dir, epoch)
                        print("")

            # samples, d_loss, g_loss = self.sess.run(
            #     [self.G_mean, self.d_loss, self.g_loss_actual],
            #     feed_dict={self.z: sample_z, self.images: zip(*sample_images)[0], self.text_emb: zip(*sample_images)[1],
            #                 self.wrong_images: fake_img_reshuffle}
            # )
            # save_images(samples, [8, 8],
            #             F.sample_dir + "/train_{:03d}.png".format(epoch))
            #if epoch % 5 == 0:
            #    iscore = self.get_inception_score()
    
    def build_model(self):
        image_dims = [self.input_height, self.input_width, self.c_dim]

        self.lr_image = tf.placeholder(
            tf.float32, [self.batch_size, 64, 64, 3], name='real_images')
        self.hr_image = tf.placeholder(
            tf.float32, [self.sample_num, 128, 128, 3], name='sample_inputs')

        self.G, self.G_ = self.generator(self.lr_image)
        self.D, self.D_logits = self.discriminator(self.hr_image, reuse=False)
        self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)

        # self.vgg_out_f = self.VGG(self.G)
        # self.vgg_out_r = self.VGG(self.hr_image)
        
        self.d_sum = histogram_summary("d", self.D)
        self.d__sum = histogram_summary("d_", self.D_)
        self.G_sum = image_summary("G", self.G)

        def sigmoid_cross_entropy_with_logits(x, y):
            try:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
            except:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

        self.d_loss_real = tf.reduce_mean(
          sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(
          sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(
          sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))
        self.mse_loss = tf.reduce_mean(tf.squared_difference(self.G, self.hr_image))

        # conv features of HR image
        self.hr_image_ = (self.hr_image + 1) * 127.5
        vgg_h = custom_vgg16(self.hr_image_, data_dict=data_dict)
        feature_h = [vgg_h.conv1_2, vgg_h.conv2_2, vgg_h.conv3_3, vgg_h.conv4_3, vgg_h.conv5_3]

        # conv features of LR Image 
        self.lr_image_ = (self.lr_image + 1) * 127.5
        vgg_l = custom_vgg16(self.lr_image_, data_dict=data_dict)
        feature_l = [vgg_l.conv1_2, vgg_l.conv2_2, vgg_l.conv3_3, vgg_l.conv4_3, vgg_l.conv5_3]

        # compute feature loss
        self.loss_f = tf.zeros(batchsize, tf.float32)
        for f, f_ in zip(vgg_h, vgg_l):
            self.loss_f += lambda_f * tf.reduce_mean(tf.sub(f, f_) ** 2, [1, 2, 3])

        self.g_loss_actual = self.g_loss + self.mse_loss # self.loss_f

        self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)
        self.g_mse_loss = scalar_summary("mse_loss", self.mse_loss)
        
        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
        self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    def train(self, config):
        """Train DCGAN"""
        if config.dataset == 'mnist':
          data_X, data_y = self.load_mnist()
        else:
          data = glob(os.path.join("./data", config.dataset, self.input_fname_pattern))
        #np.random.shuffle(data)

        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                  .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                  .minimize(self.g_loss_actual, var_list=self.g_vars)
                  
        try:
          tf.global_variables_initializer().run()
        except:
          tf.initialize_all_variables().run()

        self.g_sum = merge_summary([self.d__sum, self.G_sum, self.d_loss_fake_sum, self.g_loss_sum, self.g_mse_loss])
        self.d_sum = merge_summary(
            [self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = SummaryWriter("./logs", self.sess.graph)
        
        if config.dataset == 'mnist':
          sample_inputs = data_X[0:self.sample_num]
          sample_labels = data_y[0:self.sample_num]
        else:
          sample_files = data[0:self.sample_num]
          sample = [
              get_image(sample_file,
                        input_height=self.input_height,
                        input_width=self.input_width,
                        resize_height=self.output_height,
                        resize_width=self.output_width,
                        is_crop=self.is_crop,
                        is_grayscale=self.is_grayscale) for sample_file in sample_files]
          if (self.is_grayscale):
            sample_inputs = np.array(sample).astype(np.float32)[:, :, :, None]
          else:
            sample_inputs = np.array(sample).astype(np.float32)
      
        counter = 1
        start_time = time.time()

        if self.load(self.checkpoint_dir):
          print(" [*] Load SUCCESS")
        else:
          print(" [!] Load failed...")

        for epoch in xrange(config.epoch):
          if config.dataset == 'mnist':
            batch_idxs = min(len(data_X), config.train_size) // config.batch_size
          else:      
            data = glob(os.path.join(
              "./data", config.dataset, self.input_fname_pattern))
            batch_idxs = min(len(data), config.train_size) // config.batch_size

          for idx in xrange(0, batch_idxs):
            if config.dataset == 'mnist':
              batch_images = data_X[idx*config.batch_size:(idx+1)*config.batch_size]
              batch_labels = data_y[idx*config.batch_size:(idx+1)*config.batch_size]
            else:
              batch_files = data[idx*config.batch_size:(idx+1)*config.batch_size]
              batch = [
                  get_image(batch_file,
                            input_height=self.input_height,
                            input_width=self.input_width,
                            resize_height=self.output_height,
                            resize_width=self.output_width,
                            is_crop=self.is_crop,
                            is_grayscale=self.is_grayscale) for batch_file in batch_files]
              if (self.is_grayscale):
                batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
              else:
                batch_images = np.array(batch).astype(np.float32)

            batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
                  .astype(np.float32)

            if config.dataset == 'mnist':
              # Update D network
              _, summary_str = self.sess.run([d_optim, self.d_sum],
                feed_dict={ 
                  self.inputs: batch_images,
                  self.z: batch_z,
                  self.y:batch_labels,
                })
              self.writer.add_summary(summary_str, counter)

              # Update G network
              _, summary_str = self.sess.run([g_optim, self.g_sum],
                feed_dict={
                  self.z: batch_z, 
                  self.y:batch_labels,
                })
              self.writer.add_summary(summary_str, counter)

              # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
              _, summary_str = self.sess.run([g_optim, self.g_sum],
                feed_dict={ self.z: batch_z, self.y:batch_labels })
              self.writer.add_summary(summary_str, counter)
              
              errD_fake = self.d_loss_fake.eval({
                  self.z: batch_z, 
                  self.y:batch_labels
              })
              errD_real = self.d_loss_real.eval({
                  self.inputs: batch_images,
                  self.y:batch_labels
              })
              errG = self.g_loss.eval({
                  self.z: batch_z,
                  self.y: batch_labels
              })
            else:
              # Update D network
              _, summary_str = self.sess.run([d_optim, self.d_sum],
                feed_dict={ self.inputs: batch_images, self.z: batch_z })
              self.writer.add_summary(summary_str, counter)

              # Update G network
              _, summary_str = self.sess.run([g_optim, self.g_sum],
                feed_dict={ self.z: batch_z })
              self.writer.add_summary(summary_str, counter)

              # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
              _, summary_str = self.sess.run([g_optim, self.g_sum],
                feed_dict={ self.z: batch_z })
              self.writer.add_summary(summary_str, counter)
              
              errD_fake = self.d_loss_fake.eval({ self.z: batch_z })
              errD_real = self.d_loss_real.eval({ self.inputs: batch_images })
              errG = self.g_loss.eval({self.z: batch_z})

            counter += 1
            print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
              % (epoch, idx, batch_idxs,
                time.time() - start_time, errD_fake+errD_real, errG))

            if np.mod(counter, 100) == 1:
              if config.dataset == 'mnist':
                samples, d_loss, g_loss = self.sess.run(
                  [self.sampler, self.d_loss, self.g_loss],
                  feed_dict={
                      self.z: sample_z,
                      self.inputs: sample_inputs,
                      self.y:sample_labels,
                  }
                )
                save_images(samples, [8, 8],
                      './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
                print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss)) 
              else:
                try:
                  samples, d_loss, g_loss = self.sess.run(
                    [self.sampler, self.d_loss, self.g_loss],
                    feed_dict={
                        self.z: sample_z,
                        self.inputs: sample_inputs,
                    },
                  )
                  save_images(samples, [8, 8],
                        './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
                  print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss)) 
                except:
                  print("one pic error!...")

            if np.mod(counter, 500) == 2:
              self.save(config.checkpoint_dir, counter)

    def residual_block(self, inp, filter_dim, ind):
        rb1 = batch_norm(name='rbg_' +str(ind) + '_bn1')(conv2d(inp, filter_dim, 4, 4, 1, 1, name='rbg_' + str(ind) + '_h1_conv'))
        rb1 = tf.nn.relu(rb1)
        rb2 = batch_norm(name='rbg_' +str(ind) + '_bn2')(conv2d(rb1, filter_dim, 4, 4, 1, 1, name='rbg_' + str(ind) + '_h2_conv'))
        return tf.add(rb2, inp) # merge here

    def _phase_shift(self, I, r):
        bsize, a, b, c = I.get_shape().as_list()
        bsize = tf.shape(I)[0]  # Handling Dimension(None) type for undefined batch dim
        X = tf.reshape(I, (bsize, a, b, r, r))
        X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
        X = tf.split(1, a, X)  # a, [bsize, b, r, r]
        X = tf.concat(2, [tf.squeeze(x) for x in X])  # bsize, b, a*r, r
        X = tf.split(1, b, X)  # b, [bsize, a*r, r]
        X = tf.concat(2, [tf.squeeze(x) for x in X])  # bsize, a*r, b*r
        return tf.reshape(X, (bsize, a * r, b * r, 1))
        
        # # Helper function with main phase shift operation
        # def valueof(d, i): return d if d is not None else i 
        # p_shape = tf.shape(I)
        # bsize, a, b, c = p_shape[0], p_shape[1], p_shape[2], p_shape[3]
        # X = tf.reshape(I, (self.split_size[0], self.split_size[1], self.split_size[2], r, r))
        # X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
        # X = tf.split(X, 96, 1)  # a, [bsize, b, r, r]
        # print len(X)
        # X = tf.concat([tf.squeeze(x) for x in X], 2)  # bsize, b, a*r, r
        # X = tf.split(X, valueof(b, 96), 1)  # b, [bsize, a*r, r]
        # X = tf.concat([tf.squeeze(x) for x in X], 2)  #bsize, a*r, b*r
        # return tf.reshape(X, (valueof(bsize, 64), valueof(a, 96)*r, valueof(b, 96)*r, 1))

    def PS(self, X, r, color=False):
        # Main OP that you can arbitrarily use in you tensorflow code
        if color:
            Xc = tf.split(X, 3, 3)
            X = tf.concat([self._phase_shift(x, r) for x in Xc], 3)
        else:
            X = _phase_shift(X, r)
        return X

    # def get_output_shape_PS(self, input_shape, r):
    #     def up(d): return d if d is not None else r * d
    #     return (input_shape[0], up(input_shape[1]), up(input_shape[2]), self.c_dim)

    # def pixelShuffle(self, inp, r):
    #     out = tf.zeros(self.get_output_shape_PS(tf.shape(inp), r))
    #     print out.get_shape().as_list()
    #     for x in xrange(r): # loop across all feature maps belonging to this channel
    #         for y in xrange(r):
    #             tf.scatter_add(out[:,:,y::r,x::r], , inp[:,r*y+x::r*r,:,:])
				# print inp[:, r*y+x::r*r,:,:].get_shape().as_list()
    #     # sess.run(out.assign(out_np))
    #     return out

    def sr_model1(self, image, y=None, reuse=False):
        with tf.variable_scope("generator") as scope:
            h0 = tf.nn.relu(conv2d(image, 64, 4, 4, 1, 1, name='sr_h0_conv'))
            print (h0.get_shape())
            rb = self.residual_block(h0, 64, 1)
            rb = self.residual_block(rb, 64, 2)
            # rb = self.residual_block(rb, 64, 3)
            # rb = self.residual_block(rb, 64, 4)
            # rb = self.residual_block(rb, 64, 5)

            h2 = batch_norm(name='bnd__h2')(conv2d(rb, 12, 4, 4, 1, 1, name='sr_h2_conv'))
            # h3 = tf.add(h0, h2)

            h4 = self.PS(h2, 2, True)
            # h5 = self.upsample_pixelshuffle(h4, 256, 256, 2)

            h6 = conv2d(h4, 3, 4, 4, 1, 1, name='sr_h6_conv')
            return tf.nn.tanh(h6), h6

    def discriminator(self, image, y=None, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
            dim = 64
            h0 = lrelu(conv2d(image, dim, 4, 4, 1, 1, name='d_h0_conv'), 0.2)
            h1 = lrelu(batch_norm(name='d_bn1')(conv2d(h0, dim, 4, 4, 2, 2, name='d_h1_conv')), 0.2)

            h2 = batch_norm(name='d_bn2')(lrelu(conv2d(h1, 2*dim, 4, 4, 1, 1, name='d_h2_conv'), 0.2))
            h3 = batch_norm(name='d_bn3')(lrelu(conv2d(h2, 2*dim, 4, 4, 2, 2, name='d_h3_conv'), 0.2))
            
            h4 = batch_norm(name='d_bn4')(lrelu(conv2d(h3, 4*dim, 4, 4, 1, 1, name='d_h4_conv'), 0.2))
            h5 = batch_norm(name='d_bn5')(lrelu(conv2d(h4, 4*dim, 4, 4, 2, 2, name='d_h5_conv'), 0.2))
            
            h6 = batch_norm(name='d_bn6')(lrelu(conv2d(h5, 8*dim, 4, 4, 1, 1, name='d_h6_conv'), 0.2))
            h7 = batch_norm(name='d_bn7')(lrelu(conv2d(h6, 8*dim, 4, 4, 2, 2, name='d_h7_conv'), 0.2))

            h8 = tf.reshape(h7, [self.batch_size, -1])

            h9 = lrelu(linear(h8, self.dfc_dim, 'd_h8_lin'))
            h10 = linear(h9, 1, 'd_h9_lin')
    
            return tf.nn.sigmoid(h10), h10

    def generator(self, image, y=None):
        with tf.variable_scope("generator") as scope:
            h0 = tf.nn.relu(conv2d(image, 64, 4, 4, 1, 1, name='d_h0_conv'))

            rb = self.residual_block(h0, 64, 1)
            rb = self.residual_block(rb, 64, 2)
            rb = self.residual_block(rb, 64, 3)
            rb = self.residual_block(rb, 64, 4)
            rb = self.residual_block(rb, 64, 5)

            h2 = batch_norm(name='bnd__h2')(conv2d(rb, 64, 4, 4, 1, 1, name='d_h2_conv'))

            h3 = tf.add(h0, h2)

            h4 = self.upsample_pixelshuffle(h3, 256, 128, 1)
            h5 = self.upsample_pixelshuffle(h4, 256, 256, 2)

            h6 = conv2d(h4, 3, 4, 4, 1, 1, name='d_h6_conv')
            print (h6.get_shape())
            return tf.nn.tanh(h6), h6
      
    def save(self, checkpoint_dir, step):
        model_name = 'model'
        checkpoint_dir = os.path.join(checkpoint_dir, F.model_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
            os.path.join(checkpoint_dir, model_name),
            global_step=step)

    def make_submission(self, load_path, save_path):
        for filename in os.listdir(load_path):
            test_im = scipy.misc.imread(load_path + filename)
            sample = self.sess.run([self.SR],
                            feed_dict={self.lr_image: test_im.reshape((1,) + test_im.shape)}
                )
            scipy.misc.imsave

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, F.model_name)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
          ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
          self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
          print(" [*] Success to read {}".format(ckpt_name))
          return True
        else:
          print(" [*] Failed to find a checkpoint")
          return False
