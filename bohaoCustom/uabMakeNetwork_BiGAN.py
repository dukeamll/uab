"""
This architecture uses the following reference:
1. http://bamos.github.io/2016/08/09/deep-completion/
2. https://github.com/bamos/dcgan-completion.tensorflow
3. https://github.com/carpedm20/DCGAN-tensorflow
4. https://arxiv.org/pdf/1605.09782.pdf

Modifications to make GAN more stable for larger images:
1. Use soft and noisy labels (https://github.com/soumith/ganhacks#6-use-soft-and-noisy-labels)
2. Larger learning rate for generator, default 5 * lr_d
3. Same as https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py, run optm_g twice for each optm_d
4. Use minibatch discrimination
"""
import os
import time
import math
import imageio
import numpy as np
import tensorflow as tf
from bohaoCustom import uabMakeNetwork as network
from bohaoCustom import uabMakeNetwork_DCGAN


class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x, decay=self.momentum, updates_collections=None, epsilon=self.epsilon,
                                            scale=True, is_training=train, scope=self.name)


def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))


def image_summary(prediction):
    return (255 * (prediction / 2 + 0.5)).astype(np.uint8)


def lrelu(x, leak=0.2, name='lrelu'):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def conv2d(input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name='conv2d'):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, biases)

        return conv


def deconv2d(input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name='deconv2d', with_w=False):
    with tf.variable_scope(name):
        # filter: [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                        strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()
    with tf.variable_scope(scope or 'Linear'):
        matrix = tf.get_variable('Matrix', [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable('bias', [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias


def make_thumbnail(img_batch):
    img_num, height, width, _ = img_batch.shape
    n_row = int(np.floor(np.sqrt(img_num)))
    img = np.zeros((1, n_row*height, n_row*width, 3))
    for i in range(n_row):
        for j in range(n_row):
            img[0, (i*height):((i+1)*height), (j*width):((j+1)*width)] = img_batch[n_row*i + j]
    return img


class BiGAN(uabMakeNetwork_DCGAN.DCGAN):
    def __init__(self, inputs, trainable, input_size, model_name='', dropout_rate=None,
                 learn_rate=1e-4, decay_step=60, decay_rate=0.1, epochs=100,
                 batch_size=5, start_filter_num=32, z_dim=1000, lr_mult=5, beta1=0.5, depth=None):
        network.Network.__init__(self, inputs, trainable, dropout_rate,
                                 learn_rate, decay_step, decay_rate, epochs, batch_size)
        self.name = 'BiGAN'
        self.model_name = self.get_unique_name(model_name)
        self.sfn = start_filter_num
        self.learning_rate = None
        self.valid_d_summary = tf.placeholder(tf.float32, [])
        self.valid_g_summary = tf.placeholder(tf.float32, [])
        self.valid_iou = tf.placeholder(tf.float32, [])
        n_row = int(np.floor(np.sqrt(self.bs)))
        self.valid_images = tf.placeholder(tf.uint8, shape=[None, input_size[0] * n_row,
                                                            input_size[1] * n_row, 3], name='validation_images')
        self.class_num = 3
        self.update_ops = None
        self.z_dim = z_dim
        self.lr_mult= lr_mult
        self.beta1 = beta1

        self.output_height, self.output_width = input_size[0], input_size[1]
        if not depth:
            self.depth = int(np.log2(input_size[0] / 4))
        else:
            self.depth = depth

        # make batch normalizer
        self.d_bn = []
        self.g_bn = []
        self.e_bn = []
        for i in range(self.depth + 1):
            if i > 0:
                self.d_bn.append(batch_norm(name='d_bn{}'.format(i)))
                self.e_bn.append(batch_norm(name='e_bn{}'.format(i)))
            self.g_bn.append(batch_norm(name='g_bn{}'.format(i)))
        self.G = []
        self.E = []
        self.D, self.D_logits = [], []
        self.D_, self.D_logits_ = [], []
        self.d_loss, self.g_loss = [], []

    def encoder(self, input_):
        with tf.variable_scope('encoder'):
            h = lrelu(conv2d(input_, self.sfn, name='e_h0_conv'))
            print('e_h0: {}'.format(h.shape))
            for i in range(self.depth - 1):
                h = lrelu(self.e_bn[i](conv2d(h, self.sfn * 2 ** (i + 1), name='e_h{}_conv'.format(i + 1))))
                print('e_h{}: {}'.format(i + 1, h.shape))
            latent_height = self.output_height // (2 ** self.depth)
            h = linear(tf.reshape(h, [self.bs, latent_height * latent_height * self.sfn * 2 ** (self.depth - 1)]),
                       self.z_dim, 'e_h{}_lin'.format(self.depth + 1))
            print('e_h{}: {}'.format(self.depth, h.shape))
            self.encoded = tf.nn.tanh(h)
            return self.encoded

    def generator(self, z):
        with tf.variable_scope('generator'):
            # calculate height & width at each layer
            s_h, s_w = [self.output_height], [self.output_width]
            for i in range(self.depth):
                s_h.append(conv_out_size_same(s_h[-1], 2))
                s_w.append(conv_out_size_same(s_w[-1], 2))

            # project `z` and reshape
            print('Z: {}'.format(z.shape))
            z_, h0_w, h0_b = linear(z, self.sfn * 2 ** (self.depth - 1) * s_h[-1] * s_w[-1], 'g_h0_lin', with_w=True)
            h0 = tf.reshape(z_, [-1, s_h[-1], s_w[-1], self.sfn * 2 ** (self.depth - 1)])
            h = tf.nn.relu(self.g_bn[0](h0))
            print('g_h0: {}'.format(h0.shape))

            for i in range(1, self.depth):
                h, h_w, h_b = deconv2d(h, [self.bs, s_h[-1 - i], s_w[-1 - i], self.sfn * 2 ** (self.depth - i - 1)],
                                       name='g_h{}'.format(i), with_w=True)
                h = tf.nn.relu(self.g_bn[i](h))
                print('g_h{}(with g{}): {}'.format(i, i, h.shape))

            h, h_w, h_b = deconv2d(h, [self.bs, s_h[0], s_w[0], self.class_num], name='g_h{}'.format(self.depth + 1),
                                   with_w=True)
            print('g_h{}: {}'.format(self.depth, h.shape))

            return tf.nn.tanh(h)

    def discriminator(self, input_, encoded, reuse=False):
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()
            h = lrelu(conv2d(input_, self.sfn, name='d_h0_conv'))
            print('d_h0: {}'.format(h.shape))
            for i in range(self.depth - 1):
                h = lrelu(self.d_bn[i](conv2d(h, self.sfn * 2 ** (i + 1), name='d_h{}_conv'.format(i + 1))))
                print('d_h{}: {}'.format(i + 1, h.shape))
            latent_height = self.output_height // (2 ** self.depth)
            h = tf.reshape(h, [self.bs, latent_height * latent_height * self.sfn * 2 ** (self.depth - 1)])
            '''encoded = linear(encoded, 1000, 'd_z')
            h = tf.concat([h, encoded], axis=1)'''

            h = linear(h, 1000, 'd_x')
            encoded = linear(encoded, 1000, 'd_z')
            h = tf.concat([h, encoded], axis=1)
            print('after concat h: {}'.format(h.shape))

            h = linear(h, 1, 'd_h{}_lin'.format(self.depth + 1))
            print('d_h{}: {}'.format(self.depth, h.shape))
            return tf.nn.sigmoid(h), h

    def create_graph(self, x_name, class_num, start_filter_num=32, reduce_dim=True):
        self.class_num = class_num
        print('Make Gnerator:')
        self.G = self.generator(self.inputs['Z'])
        print('Make Encoder:')
        self.E = self.encoder(self.inputs[x_name])
        print('Make Discriminator:')
        self.D, self.D_logits = self.discriminator(self.inputs[x_name], self.E, reuse=False)
        self.D_, self.D_logits_ = self.discriminator(self.G, self.inputs['Z'], reuse=True)

    def make_optimizer(self, train_var_filter):
        with tf.control_dependencies(self.update_ops):
            t_vars = tf.trainable_variables()
            d_vars = [var for var in t_vars if 'd_' in var.name or 'e_' in var.name]
            g_vars = [var for var in t_vars if 'g_' in var.name]
            optm_d = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1).\
                minimize(self.d_loss, var_list=d_vars, global_step=self.global_step)
            optm_g = tf.train.AdamOptimizer(self.learning_rate * self.lr_mult, beta1=self.beta1).\
                minimize(self.g_loss, var_list=g_vars, global_step=self.global_step)
            self.optimizer = {'d': optm_d, 'g': optm_g}

    def test(self, x_name, sess, test_iterator):
        result = []
        for X_batch in test_iterator:
            pred = sess.run(self.encoded, feed_dict={self.inputs[x_name]: X_batch,
                                                     self.trainable: False})
            result.append(pred)
        result = np.vstack(result)
        return result

    def encoding(self, x_name, sess, test_iterator):
        for X_batch in test_iterator:
            pred = sess.run(self.encoded, feed_dict={self.inputs[x_name]: X_batch,
                                                     self.trainable: False})
            yield pred[0, :]
