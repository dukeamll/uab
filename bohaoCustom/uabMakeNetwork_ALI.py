import os
import time
import imageio
import collections
import numpy as np
import tensorflow as tf
from bohaoCustom import uabMakeNetwork as network
from bohaoCustom import uabMakeNetwork_DCGAN


def batch_norm(h, training=True):
    return tf.contrib.layers.batch_norm(h, is_training=training, updates_collections=None, decay=0.9,
                                        epsilon=1e-5, scale=True,)


def lrelu(x, slope=0.1):
    return tf.maximum(x, slope * x)


def conv_maxout(x, num_pieces=2):
    splited = tf.split(x, num_pieces, axis=3)
    h = tf.stack(splited, axis=-1)
    h = tf.reduce_max(h, axis=4)
    return h


def add_nontied_bias(x, initializer=None):
    with tf.variable_scope('add_nontied_bias'):
        if initializer is not None:
            bias = tf.get_variable('bias', shape=x.get_shape().as_list()[1:], trainable=True, initializer=initializer)
        else:
            bias = tf.get_variable('bias', shape=x.get_shape().as_list()[1:], trainable=True,
                                   initializer=tf.zeros_initializer())
        output = x + bias
    return output

def cal_marginal(raw_marginals, eps=1e-7):
    marg = np.clip(raw_marginals.mean(axis=0), eps, 1. - eps)
    return np.log(marg / (1. - marg))


class ALI(uabMakeNetwork_DCGAN.DCGAN):
    def __init__(self, inputs, trainable, input_size, model_name='', dropout_rate=None,
                 learn_rate=1e-4, decay_step=60, decay_rate=0.1, epochs=100,
                 batch_size=5, start_filter_num=32, z_dim=1000, lr_mult=5, beta1=0.5, raw_marginal=None):
        network.Network.__init__(self, inputs, trainable, dropout_rate,
                                 learn_rate, decay_step, decay_rate, epochs, batch_size)
        self.name = 'ALI'
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

        if raw_marginal is not None:
            marginal = cal_marginal(raw_marginal)
            self.marginal_initializer = tf.constant_initializer(marginal, tf.float32)
        else:
            self.marginal_initializer = None

        self.w_init = lambda: tf.truncated_normal_initializer(mean=0.0, stddev=0.001)
        self.slope = 0.2
        self.train_g = tf.placeholder(tf.bool, [])
        self.train_d = tf.placeholder(tf.bool, [])

    def generator_x(self, input_, train=True, reuse=False):
        with tf.variable_scope('generator_x', reuse=reuse):
            h = input_
            with tf.variable_scope('block0'):
                h = tf.layers.conv2d(h, filters=2048, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                                     use_bias=False, kernel_initializer=self.w_init())
                h = batch_norm(h, training=train)
                h = lrelu(h, slope=self.slope)
            with tf.variable_scope('block1'):
                h = tf.layers.conv2d(h, filters=256, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=False,
                         kernel_initializer=self.w_init())
                h = batch_norm(h, training=train)
                h = lrelu(h, slope=self.slope)
            Block = collections.namedtuple('Block', ['i', 'filter_num', 'kernel_size', 'stride'])
            blocks = [
                Block(2, 256, 4, 1),
                Block(3, 128, 4, 2),
                Block(4, 128, 4, 1),
                Block(5, 64, 4, 2),
                Block(6, 64, 4, 1),
                Block(7, 64, 4, 2),
            ]
            for i, b in enumerate(blocks):
                with tf.variable_scope('block{}'.format(b.i)):
                    h = tf.layers.conv2d_transpose(h, filters=b.filter_num, kernel_size=(b.kernel_size, b.kernel_size),
                                                   strides=(b.stride, b.stride), padding='valid', use_bias=False,
                                                   kernel_initializer=self.w_init())
                    h = batch_norm(h, training=train)
                    h = lrelu(h, slope=self.slope)
            with tf.variable_scope('block8'):
                h = tf.layers.conv2d(h, filters=3, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=False,
                                     kernel_initializer=self.w_init())
                h = add_nontied_bias(h, initializer=self.marginal_initializer)
                h = tf.nn.sigmoid(h)
            output = h
        return output

    def generator_z(self, input_, train=True, reuse=False):
        self.input = input_
        with tf.variable_scope('generator_z', reuse=reuse):
            h = self.input
            Block = collections.namedtuple('Block', ['i', 'filter_num', 'kernel_size', 'stride'])
            blocks = [
                Block(0, 64, 4, 2),
                Block(1, 64, 4, 1),
                Block(2, 128, 4, 2),
                Block(3, 128, 4, 1),
                Block(4, 256, 4, 2),
                Block(5, 256, 4, 1),
                Block(6, 2048, 1, 1),
                Block(7, 2048, 1, 1),
            ]
            for b in blocks:
                with tf.variable_scope('block{}'.format(b.i)):
                    h = tf.layers.conv2d(h, filters=b.filter_num, kernel_size=(b.kernel_size, b.kernel_size),
                             strides=(b.stride, b.stride), padding='valid', use_bias=False,
                                         kernel_initializer=self.w_init())
                    h = batch_norm(h, training=train)
                    h = lrelu(h, slope=self.slope)
            with tf.variable_scope('block8'):
                with tf.variable_scope('mu'):
                    h_mu = tf.layers.conv2d(h, filters=self.z_dim, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                                use_bias=False, kernel_initializer=self.w_init())
                    h_mu = add_nontied_bias(h_mu)
                    self.G_z_mu = h_mu
                with tf.variable_scope('sigma'):
                    h_sigma = tf.layers.conv2d(h, filters=self.z_dim, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                                   use_bias=False, kernel_initializer=self.w_init())
                    h_sigma = add_nontied_bias(h_sigma)
                    self.G_z_sigma = h_sigma
                    h_sigma = tf.exp(h_sigma)
                rng = tf.random_normal(shape=tf.shape(h_mu))
                output = (rng * h_sigma) + h_mu
        return output

    def discriminator(self, input_x, input_z, train=True, reuse=False):
        with tf.variable_scope('discriminator', reuse=reuse):
            h_x = input_x
            h_z = input_z
            with tf.variable_scope('x'):
                Block = collections.namedtuple('Block', ['i', 'filter_num', 'kernel_size', 'stride', 'is_bn'])
                blocks = [
                    Block(0, 64, 4, 2, False),
                    Block(1, 64, 4, 1, True),
                    Block(2, 128, 4, 2, True),
                    Block(3, 128, 4, 1, True),
                    Block(4, 256, 4, 2, True),
                    Block(5, 256, 4, 1, True),
                ]
                for b in blocks:
                    with tf.variable_scope('block{}'.format(b.i)):
                        h_x = tf.layers.dropout(h_x, rate=0.2, training=train)
                        h_x = tf.layers.conv2d(h_x, filters=b.filter_num, kernel_size=(b.kernel_size, b.kernel_size),
                                               strides=(b.stride, b.stride), padding='valid', use_bias=not b.is_bn,
                                               kernel_initializer=self.w_init())
                        if b.is_bn:
                            h_x = batch_norm(h_x, training=train)
                        h_x = lrelu(h_x, slope=self.slope)
            with tf.variable_scope('z'):
                with tf.variable_scope('block0'):
                    h_z = tf.layers.dropout(h_z, rate=0.2, training=train)
                    h_z = tf.layers.conv2d(h_z, filters=2048, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                                           use_bias=True, kernel_initializer=self.w_init())
                    h_z = lrelu(h_z, slope=self.slope)
                with tf.variable_scope('block1'):
                    h_z = tf.layers.dropout(h_z, rate=0.2, training=train)
                    h_z = tf.layers.conv2d(h_z, filters=2048, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                                           use_bias=True, kernel_initializer=self.w_init())
                    h_z = lrelu(h_z, slope=self.slope)
            with tf.variable_scope('xz'):
                h_xz = tf.concat([h_x, h_z], axis=h_x.get_shape().ndims - 1)
                with tf.variable_scope('block0'):
                    h_xz = tf.layers.dropout(h_xz, rate=0.2, training=train)
                    h_xz = tf.layers.conv2d(h_xz, filters=4096, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                                            use_bias=True, kernel_initializer=self.w_init())
                    h_xz = lrelu(h_xz, slope=self.slope)

                with tf.variable_scope('block1'):
                    h_xz = tf.layers.dropout(h_xz, rate=0.2, training=train)
                    h_xz = tf.layers.conv2d(h_xz, filters=4096, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                                            use_bias=True, kernel_initializer=self.w_init())
                    h_xz = lrelu(h_xz, slope=self.slope)

                with tf.variable_scope('block2'):
                    h_xz = tf.layers.dropout(h_xz, rate=0.2, training=train)
                    h_xz = tf.layers.conv2d(h_xz, filters=1, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                                            use_bias=True, kernel_initializer=self.w_init())
            logits = h_xz
            output = tf.nn.sigmoid(h_xz)
            return output, logits

    def create_graph(self, x_name, class_num, start_filter_num=32, reduce_dim=True):
        self.class_num = class_num
        print('Make Gnerator:')
        self.G_x = self.generator_x(self.inputs['Z'], train=self.train_g, reuse=False)
        self.G_z = self.generator_z(self.inputs[x_name], train=self.train_g, reuse=False)
        print('Make Discriminator:')
        self.D, self.D_logits = self.discriminator(self.inputs[x_name], self.G_z, train=self.train_d, reuse=False)
        self.D_, self.D_logits_ = self.discriminator(self.G_x, self.inputs['Z'], train=self.train_d, reuse=True)
        resampler = self.generator_x(self.generator_z(self.inputs[x_name], train=False, reuse=True), train=False,
                                     reuse=True)
        self.resampler = tf.concat([self.inputs[x_name], resampler], axis=2)

    def make_loss(self, z_name, loss_type='xent', **kwargs):
        with tf.variable_scope('d_loss'):
            self.d_loss = tf.reduce_mean(tf.nn.softplus(-self.D_logits) + tf.nn.softplus(self.D_logits_))
        with tf.variable_scope('g_loss'):
            self.g_loss = tf.reduce_mean(tf.nn.softplus(self.D_logits) + tf.nn.softplus(-self.D_logits_))

    def make_optimizer(self, train_var_filter):
        with tf.control_dependencies(self.update_ops):
            t_vars = tf.trainable_variables()
            d_vars = [var for var in t_vars if 'discriminator' in var.name]
            g_vars = [var for var in t_vars if 'generator_x' in var.name] + \
                     [var for var in t_vars if 'generator_z' in var.name]
            optm_d = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1).\
                minimize(self.d_loss, var_list=d_vars, global_step=self.global_step)
            optm_g = tf.train.AdamOptimizer(self.learning_rate * self.lr_mult, beta1=self.beta1).\
                minimize(self.g_loss, var_list=g_vars, global_step=self.global_step)
            self.optimizer = {'d': optm_d, 'g': optm_g}

    def train(self, x_name, z_name, n_train, sess, summary_writer, n_valid=1000,
              train_reader=None, valid_reader=None,
              image_summary=None, verb_step=100, save_epoch=5,
              img_mean=np.array((0, 0, 0), dtype=np.float32),
              continue_dir=None, valid_iou=False):
        # define summary operations
        valid_g_summary_op = tf.summary.scalar('g_loss_validation', self.valid_g_summary)
        valid_d_summary_op = tf.summary.scalar('d_loss_validation', self.valid_d_summary)
        valid_image_summary_op = tf.summary.image('Validation_images_summary', self.valid_images,
                                                  max_outputs=10)

        if continue_dir is not None:
            self.load(continue_dir, sess)
            gs = sess.run(self.global_step)
            start_epoch = int(np.ceil(gs/n_train*self.bs))
            start_step = gs - int(start_epoch*n_train/self.bs)
        else:
            start_epoch = 0
            start_step = 0

        loss_valid_min = np.inf
        for epoch in range(start_epoch, self.epochs):
            start_time = time.time()
            for step_cnt, step in enumerate(range(start_step, n_train, self.bs)):
                X_batch, _ = train_reader.readerAction(sess)
                Z_batch = np.random.normal(size=(self.bs, 1, 1, self.z_dim)).astype(np.float32)
                X_batch = X_batch / 255.
                _, _, self.global_step_value = sess.run([self.optimizer['d'], self.optimizer['g'], self.global_step],
                                                        feed_dict={self.inputs[x_name]: X_batch,
                                                                   self.inputs[z_name]: Z_batch,
                                                                   self.train_g: True, self.train_d: True})

                if step_cnt % verb_step == 0:
                    d_loss, g_loss, step_summary = sess.run([self.d_loss, self.g_loss, self.summary],
                                                            feed_dict={self.inputs[x_name]: X_batch,
                                                                       self.inputs[z_name]: Z_batch,
                                                                       self.train_g: False, self.train_d: False})
                    summary_writer.add_summary(step_summary, self.global_step_value)
                    print('Epoch {:d} step {:d}\td_loss = {:.3f}, g_loss = {:.3f}'.
                          format(epoch, step_cnt, d_loss, g_loss))
            # validation
            loss_valid_mean = []
            g_loss_val_mean = []
            d_loss_val_mean = []
            for step in range(0, n_valid, self.bs):
                X_batch_val, _ = valid_reader.readerAction(sess)
                Z_batch_val = np.random.normal(size=(self.bs, 1, 1, self.z_dim)).astype(np.float32)
                X_batch_val = X_batch_val / 255.
                d_loss_val, g_loss_val = sess.run([self.d_loss, self.g_loss],
                                                  feed_dict={self.inputs[x_name]: X_batch_val,
                                                             self.inputs[z_name]: Z_batch_val,
                                                             self.train_g: False, self.train_d: False})
                loss_valid_mean.append(d_loss_val+g_loss_val)
                g_loss_val_mean.append(g_loss_val)
                d_loss_val_mean.append(d_loss_val)
            loss_valid_mean = np.mean(loss_valid_mean)
            duration = time.time() - start_time
            print('Validation loss: {:.3f}, duration: {:.3f}'.format(loss_valid_mean, duration))
            valid_g_summary = sess.run(valid_g_summary_op,
                                       feed_dict={self.valid_g_summary: np.mean(g_loss_val_mean)})
            valid_d_summary = sess.run(valid_d_summary_op,
                                       feed_dict={self.valid_d_summary: np.mean(d_loss_val_mean)})
            summary_writer.add_summary(valid_g_summary, self.global_step_value)
            summary_writer.add_summary(valid_d_summary, self.global_step_value)
            if loss_valid_mean < loss_valid_min:
                loss_valid_min = loss_valid_mean
                saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)
                saver.save(sess, '{}/best_model.ckpt'.format(self.ckdir))

            Z_batch_sample = np.random.normal(size=(self.bs, 1, 1, self.z_dim)).astype(np.float32)
            valid_img_gen = sess.run(self.G_x, feed_dict={self.inputs[z_name]: Z_batch_sample,
                                                          self.train_g: False, self.train_d: False})
            valid_img_gen = uabMakeNetwork_DCGAN.make_thumbnail(valid_img_gen)

            thumbnail_path = self.ckdir + '_thumb'
            if not os.path.exists(thumbnail_path):
                os.makedirs(thumbnail_path)
            imageio.imsave(os.path.join(thumbnail_path, '{}.png'.format(epoch)),
                           (valid_img_gen[0, :, :, :] * 255).astype(np.uint8))

            if image_summary is not None:
                valid_image_summary = sess.run(valid_image_summary_op,
                                               feed_dict={self.valid_images:
                                                              image_summary(valid_img_gen)})
                summary_writer.add_summary(valid_image_summary, self.global_step_value)

            if epoch % save_epoch == 0:
                saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)
                saver.save(sess, '{}/model_{}.ckpt'.format(self.ckdir, epoch), global_step=self.global_step)

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
