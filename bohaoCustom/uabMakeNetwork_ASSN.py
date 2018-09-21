import os
import time
import numpy as np
import tensorflow as tf
import uabUtilreader
import util_functions
from bohaoCustom import uabMakeNetwork_UNet
from bohaoCustom import uabMakeNetwork as network


class SSAN(uabMakeNetwork_UNet.UnetModelGAN_V4RGB):
    def __init__(self, inputs, trainable, input_size, model_name='', dropout_rate=None,
                 learn_rate=1e-4, decay_step=60, decay_rate=0.1, epochs=100,
                 batch_size=5, start_filter_num=32, lada=2, w_pretrained=None):
        network.Network.__init__(self, inputs, trainable, dropout_rate,
                                 learn_rate, decay_step, decay_rate, epochs, batch_size)
        self.lr = self.make_list(learn_rate)
        self.ds = self.make_list(decay_step)
        self.dr = self.make_list(decay_rate)
        self.name = 'SSAN'
        self.model_name = self.get_unique_name(model_name)
        self.sfn = start_filter_num
        self.lada = lada
        self.w_pretrained = w_pretrained
        self.input_size = input_size
        self.learning_rate = None
        self.valid_cross_entropy = tf.placeholder(tf.float32, [])
        self.valid_iou = tf.placeholder(tf.float32, [])
        self.valid_d_loss = tf.placeholder(tf.float32, [])
        self.valid_g_loss = tf.placeholder(tf.float32, [])
        self.valid_images = tf.placeholder(
            tf.uint8, shape=[None, input_size[0] - self.get_overlap(), (input_size[1] - self.get_overlap()) * 3, 3],
            name='validation_images')
        self.update_ops = None
        self.config = None
        self.hard_label = None
        self.fake_logit = None
        self.true_logit = None
        self.d_loss = None
        self.g_loss = None

    def get_overlap(self):
        return 0

    def conv(self, name, input, strides, padding, add_bias, apply_relu, atrous_rate=None):
        """
        Helper function for loading convolution weights from weight dictionary.
        """
        with tf.variable_scope(name):

            # Load kernel weights and apply convolution
            w_kernel = self.w_pretrained[name + '/kernel:0']
            w_kernel = tf.Variable(initial_value=w_kernel, trainable=True)

            if not atrous_rate:
                conv_out = tf.nn.conv2d(input, w_kernel, strides, padding)
            else:
                conv_out = tf.nn.atrous_conv2d(input, w_kernel, atrous_rate, padding)
            if add_bias:
                # Load bias values and add them to conv output
                w_bias = self.w_pretrained[name + '/bias:0']
                w_bias = tf.Variable(initial_value=w_bias, trainable=True)
                conv_out = tf.nn.bias_add(conv_out, w_bias)

            if apply_relu:
                # Apply ReLu nonlinearity
                conv_out = tf.nn.relu(conv_out)

        return conv_out

    def make_encoder(self, x_name):
        with tf.variable_scope('encoder'):
            h = self.conv('conv1_1', self.inputs[x_name], strides=[1, 1, 1, 1], padding='VALID', add_bias=True,
                          apply_relu=True)
            h = self.conv('conv1_2', h, strides=[1, 1, 1, 1], padding='VALID', add_bias=True, apply_relu=True)
            h = tf.layers.max_pooling2d(h, pool_size=(2, 2), strides=(2, 2), padding='valid', name='pool1')

            h = self.conv('conv2_1', h, strides=[1, 1, 1, 1], padding='VALID', add_bias=True, apply_relu=True)
            h = self.conv('conv2_2', h, strides=[1, 1, 1, 1], padding='VALID', add_bias=True, apply_relu=True)
            h = tf.layers.max_pooling2d(h, pool_size=(2, 2), strides=(2, 2), padding='valid', name='pool2')

            h = self.conv('conv3_1', h, strides=[1, 1, 1, 1], padding='VALID', add_bias=True, apply_relu=True)
            h = self.conv('conv3_2', h, strides=[1, 1, 1, 1], padding='VALID', add_bias=True, apply_relu=True)
            h = self.conv('conv3_3', h, strides=[1, 1, 1, 1], padding='VALID', add_bias=True, apply_relu=True)
            h = tf.layers.max_pooling2d(h, pool_size=(2, 2), strides=(2, 2), padding='valid', name='pool3')

            h = self.conv('conv4_1', h, strides=[1, 1, 1, 1], padding='VALID', add_bias=True, apply_relu=True)
            h = self.conv('conv4_2', h, strides=[1, 1, 1, 1], padding='VALID', add_bias=True, apply_relu=True)
            h = self.conv('conv4_3', h, strides=[1, 1, 1, 1], padding='VALID', add_bias=True, apply_relu=True)

            h = self.conv('conv5_1', h, strides=[1, 1, 1, 1], padding='VALID', add_bias=True, apply_relu=True,
                          atrous_rate=2)
            h = self.conv('conv5_2', h, strides=[1, 1, 1, 1], padding='VALID', add_bias=True, apply_relu=True,
                          atrous_rate=2)
            h = self.conv('conv5_3', h, strides=[1, 1, 1, 1], padding='VALID', add_bias=True, apply_relu=True,
                          atrous_rate=2)
            h = self.conv('fc6', h, strides=[1, 1, 1, 1], padding='VALID', add_bias=True, apply_relu=True,
                          atrous_rate=4)

            h = tf.layers.dropout(h, rate=0.5, name='drop6')
            h = self.conv('fc7', h, strides=[1, 1, 1, 1], padding='VALID', add_bias=True, apply_relu=True)
            h = tf.layers.dropout(h, rate=0.5, name='drop7')
            h = self.conv('final', h, strides=[1, 1, 1, 1], padding='VALID', add_bias=True, apply_relu=True)

            h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT', name='ctx_pad1_1')
            h = self.conv('ctx_conv1_1', h, strides=[1, 1, 1, 1], padding='VALID', add_bias=True, apply_relu=True)
            h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT', name='ctx_pad1_2')
            h = self.conv('ctx_conv1_2', h, strides=[1, 1, 1, 1], padding='VALID', add_bias=True, apply_relu=True)

            h = tf.pad(h, [[0, 0], [2, 2], [2, 2], [0, 0]], mode='CONSTANT', name='ctx_pad2_1')
            h = self.conv('ctx_conv2_1', h, strides=[1, 1, 1, 1], padding='VALID', add_bias=True, apply_relu=True,
                          atrous_rate=2)

            h = tf.pad(h, [[0, 0], [4, 4], [4, 4], [0, 0]], mode='CONSTANT', name='ctx_pad3_1')
            h = self.conv('ctx_conv3_1', h, strides=[1, 1, 1, 1], padding='VALID', add_bias=True, apply_relu=True,
                          atrous_rate=4)

            h = tf.pad(h, [[0, 0], [8, 8], [8, 8], [0, 0]], mode='CONSTANT', name='ctx_pad4_1')
            h = self.conv('ctx_conv4_1', h, strides=[1, 1, 1, 1], padding='VALID', add_bias=True, apply_relu=True,
                          atrous_rate=8)

            h = tf.pad(h, [[0, 0], [16, 16], [16, 16], [0, 0]], mode='CONSTANT', name='ctx_pad5_1')
            h = self.conv('ctx_conv5_1', h, strides=[1, 1, 1, 1], padding='VALID', add_bias=True, apply_relu=True,
                          atrous_rate=16)

            h = tf.pad(h, [[0, 0], [32, 32], [32, 32], [0, 0]], mode='CONSTANT', name='ctx_pad6_1')
            h = self.conv('ctx_conv6_1', h, strides=[1, 1, 1, 1], padding='VALID', add_bias=True, apply_relu=True,
                          atrous_rate=32)

            h = tf.pad(h, [[0, 0], [64, 64], [64, 64], [0, 0]], mode='CONSTANT', name='ctx_pad7_1')
            h = self.conv('ctx_conv7_1', h, strides=[1, 1, 1, 1], padding='VALID', add_bias=True, apply_relu=True,
                          atrous_rate=64)

            h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT', name='ctx_pad_fc1')
            h = self.conv('ctx_fc1', h, strides=[1, 1, 1, 1], padding='VALID', add_bias=True, apply_relu=True)
            h = self.conv('ctx_final', h, strides=[1, 1, 1, 1], padding='VALID', add_bias=True, apply_relu=False)

            h = tf.image.resize_bilinear(h, size=(self.input_size[0], self.input_size[1]))
            logits = self.conv('ctx_upsample', h, strides=[1, 1, 1, 1], padding='SAME', add_bias=False, apply_relu=True)
        return logits

    def make_discriminator(self, y, sfn=4, reuse=False):
        with tf.variable_scope('discr', reuse=reuse):
            # downsample
            conv1, pool1 = self.conv_conv_pool(y, [96, 128, 128], self.trainable, name='conv1', kernal_size=(3, 3),
                                               conv_stride=(1, 1), padding='valid', dropout=self.dropout_rate,
                                               pool=True, bn=False, activation=tf.nn.relu)
            conv2, pool2 = self.conv_conv_pool(pool1, [128, 128], self.trainable, name='conv2', kernal_size=(3, 3),
                                               conv_stride=(1, 1), padding='valid', dropout=self.dropout_rate,
                                               pool=True, bn=False, activation=tf.nn.relu)
            conv3 = self.conv_conv_pool(pool2, [256], self.trainable, name='conv3', kernal_size=(3, 3),
                                        conv_stride=(1, 1), padding='valid', dropout=self.dropout_rate, pool=False,
                                        bn=True, activation=tf.nn.relu)
            conv4 = tf.layers.conv2d(conv3, 2, kernel_size=(3, 3), dilation_rate=(32, 32), name='layerconv4')
            flat = tf.reshape(conv4, shape=[self.bs, 50*50*2])
            return self.fc_fc(flat, [1], self.trainable, name='fc_final', activation=None, dropout=False)

    def create_graph(self, names, class_num, start_filter_num=32):
        self.class_num = class_num

        conv9 = self.make_encoder(names[0])
        self.pred = tf.layers.conv2d(conv9, class_num, (1, 1), name='final', activation=None, padding='same')
        self.output = tf.nn.softmax(self.pred)

        with tf.variable_scope('Discriminator'):
            true_y = tf.cast(self.inputs[names[1]], tf.float32)
            orig_rgb = self.inputs[names[0]] * true_y
            pred_rgb = self.inputs[names[0]] * tf.expand_dims(self.output[:, :, :, 1], axis=-1)
            self.true_logit = self.make_discriminator(orig_rgb, sfn=start_filter_num//4, reuse=False)
            self.fake_logit = self.make_discriminator(pred_rgb, sfn=start_filter_num//4, reuse=True)

    def make_loss(self, y_name, loss_type='xent', **kwargs):
        with tf.variable_scope('loss'):
            pred_flat = tf.reshape(self.pred, [-1, self.class_num])
            y_flat = tf.reshape(tf.squeeze(self.inputs[y_name], axis=[3]), [-1, ])
            indices = tf.squeeze(tf.where(tf.less_equal(y_flat, self.class_num - 1)), 1)
            gt = tf.gather(y_flat, indices)
            prediction = tf.gather(pred_flat, indices)

            pred = tf.argmax(prediction, axis=-1, output_type=tf.int32)
            intersect = tf.cast(tf.reduce_sum(gt * pred), tf.float32)
            union = tf.cast(tf.reduce_sum(gt), tf.float32) + tf.cast(tf.reduce_sum(pred), tf.float32) \
                    - tf.cast(tf.reduce_sum(gt * pred), tf.float32)
            self.loss_iou = tf.convert_to_tensor([intersect, union])
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt))

        with tf.variable_scope('adv_loss'):
            d_loss_fake_0 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_logit,
                                                                                   labels=tf.zeros([self.bs, 1])))
            d_loss_fake_1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_logit,
                                                                                   labels=tf.ones([self.bs, 1])))
            d_loss_real_1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.true_logit,
                                                                                   labels=tf.ones([self.bs, 1])))
            self.g_loss = self.lada * d_loss_fake_1 + self.loss
            self.d_loss = d_loss_fake_0 + d_loss_real_1

    def make_optimizer(self, train_var_filter):
        with tf.control_dependencies(self.update_ops):
            if train_var_filter is None:
                t_vars = tf.trainable_variables()
                e_vars = [var for var in t_vars if 'encoder' in var.name]
                d_vars = [var for var in t_vars if 'Discriminator' in var.name]
                g_optm = tf.train.AdamOptimizer(self.learning_rate[0], name='Adam_g').\
                    minimize(self.g_loss, var_list=e_vars, global_step=self.global_step)
                d_optm = tf.train.AdamOptimizer(self.learning_rate[1], name='Adam_d').\
                    minimize(self.d_loss, var_list=d_vars, global_step=None)
                self.optimizer = [g_optm, d_optm]

    def make_learning_rate(self, n_train):
        self.learning_rate = []
        for i in range(2):
            self.learning_rate.append(tf.train.exponential_decay(self.lr[i], self.global_step,
                                                                 tf.cast(n_train/self.bs * self.ds[i], tf.int32),
                                                                 self.dr[i], staircase=True))

    def make_update_ops(self, x_name, y_name):
        tf.add_to_collection('inputs', self.inputs[x_name])
        tf.add_to_collection('inputs', self.inputs[y_name])
        tf.add_to_collection('outputs', self.pred)
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    def train(self, x_name, y_name, n_train, sess, summary_writer, n_valid=1000,
              train_reader=None, train_reader_source=None, train_reader_target=None, valid_reader=None,
              image_summary=None, verb_step=100, save_epoch=5,
              img_mean=np.array((0, 0, 0), dtype=np.float32),
              continue_dir=None, valid_iou=False):
        # define summary operations
        valid_cross_entropy_summary_op = tf.summary.scalar('xent_validation', self.valid_cross_entropy)
        valid_iou_summary_op = tf.summary.scalar('iou_validation', self.valid_iou)
        valid_d_loss_summary_op = tf.summary.scalar('d_loss_validation', self.valid_d_loss)
        valid_g_loss_summary_op = tf.summary.scalar('g_loss_validation', self.valid_g_loss)
        valid_image_summary_op = tf.summary.image('Validation_images_summary', self.valid_images,
                                                  max_outputs=10)

        if continue_dir is not None and os.path.exists(continue_dir):
            self.load(continue_dir, sess)
            gs = sess.run(self.global_step)
            start_epoch = int(np.ceil(gs/n_train*self.bs))
            start_step = gs - int(start_epoch*n_train/self.bs)
        else:
            start_epoch = 0
            start_step = 0

        cross_entropy_valid_min = np.inf
        iou_valid_max = 0
        for epoch in range(start_epoch, self.epochs):
            start_time = time.time()
            for step in range(start_step, n_train, self.bs):
                X_batch, y_batch = train_reader.readerAction(sess)
                _, self.global_step_value = sess.run([self.optimizer[0], self.global_step],
                                                     feed_dict={self.inputs[x_name]:X_batch,
                                                                self.inputs[y_name]:y_batch,
                                                                self.trainable: True})
                X_batch, _ = train_reader_target.readerAction(sess)
                _, y_batch = train_reader_source.readerAction(sess)
                sess.run([self.optimizer[1]], feed_dict={self.inputs[x_name]: X_batch,
                                                         self.inputs[y_name]: y_batch,
                                                         self.trainable: True})

                if self.global_step_value % verb_step == 0:
                    step_cross_entropy, step_summary = sess.run([self.loss, self.summary],
                                                                feed_dict={self.inputs[x_name]: X_batch,
                                                                           self.inputs[y_name]: y_batch,
                                                                           self.trainable: False})
                    summary_writer.add_summary(step_summary, self.global_step_value)
                    print('Epoch {:d} step {:d}\tcross entropy = {:.3f}'.
                          format(epoch, self.global_step_value, step_cross_entropy))
            # validation
            cross_entropy_valid_mean = []
            d_loss_valid_mean = []
            g_loss_valid_mean = []
            iou_valid_mean = np.zeros(2)
            X_batch_val, y_batch_val, pred_valid = None, None, None
            for step in range(0, n_valid, self.bs):
                X_batch_val, y_batch_val = valid_reader.readerAction(sess)
                pred_valid, cross_entropy_valid, iou_valid = sess.run(
                    [self.pred, self.loss, self.loss_iou], feed_dict={self.inputs[x_name]: X_batch_val,
                                                                      self.inputs[y_name]: y_batch_val,
                                                                      self.trainable: False})
                _, y_batch_val_target = train_reader_source.readerAction(sess)
                d_loss_valid, g_loss_valid = sess.run([self.d_loss, self.g_loss],
                                                      feed_dict={self.inputs[x_name]: X_batch_val,
                                                                 self.inputs[y_name]: y_batch_val_target,
                                                                 self.trainable: False})
                cross_entropy_valid_mean.append(cross_entropy_valid)
                d_loss_valid_mean.append(d_loss_valid)
                g_loss_valid_mean.append(g_loss_valid)
                iou_valid_mean += iou_valid
            cross_entropy_valid_mean = np.mean(cross_entropy_valid_mean)
            d_loss_valid_mean = np.mean(d_loss_valid_mean)
            g_loss_valid_mean = np.mean(g_loss_valid_mean)
            iou_valid_mean = iou_valid_mean[0] / iou_valid_mean[1]
            duration = time.time() - start_time
            if valid_iou:
                print('Validation IoU: {:.3f}, duration: {:.3f}'.format(iou_valid_mean, duration))
            else:
                print('Val xent: {:.3f}, g_loss: {:.3f}, d_loss: {:.3f}, duration: {:.3f}'.
                      format(cross_entropy_valid_mean, d_loss_valid_mean, g_loss_valid_mean, duration))
            valid_summaries = sess.run([valid_cross_entropy_summary_op, valid_iou_summary_op,
                                        valid_d_loss_summary_op, valid_g_loss_summary_op],
                                       feed_dict={self.valid_cross_entropy: cross_entropy_valid_mean,
                                                  self.valid_iou: iou_valid_mean,
                                                  self.valid_d_loss: d_loss_valid_mean,
                                                  self.valid_g_loss: g_loss_valid_mean})
            for i in range(4):
                summary_writer.add_summary(valid_summaries[i], self.global_step_value)
            if valid_iou:
                if iou_valid_mean > iou_valid_max:
                    iou_valid_max = iou_valid_mean
                    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)
                    saver.save(sess, '{}/best_model.ckpt'.format(self.ckdir))

            else:
                if cross_entropy_valid_mean < cross_entropy_valid_min:
                    cross_entropy_valid_min = cross_entropy_valid_mean
                    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)
                    saver.save(sess, '{}/best_model.ckpt'.format(self.ckdir))

            if image_summary is not None:
                valid_image_summary = sess.run(
                    valid_image_summary_op, feed_dict={
                        self.valid_images: util_functions.image_summary(X_batch_val, y_batch_val, pred_valid, img_mean)})
                summary_writer.add_summary(valid_image_summary, self.global_step_value)

            if epoch % save_epoch == 0:
                saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)
                saver.save(sess, '{}/model_{}.ckpt'.format(self.ckdir, epoch), global_step=self.global_step)


class SSAN_UNet(SSAN):
    def __init__(self, inputs, trainable, input_size, model_name='', dropout_rate=None,
                 learn_rate=1e-4, decay_step=60, decay_rate=0.1, epochs=100,
                 batch_size=5, start_filter_num=32, lada=2, slow_iter=500):
        network.Network.__init__(self, inputs, trainable, dropout_rate,
                                 learn_rate, decay_step, decay_rate, epochs, batch_size)
        self.lr = self.make_list(learn_rate)
        self.ds = self.make_list(decay_step)
        self.dr = self.make_list(decay_rate)
        self.name = 'SSAN_Unet'
        self.model_name = self.get_unique_name(model_name)
        self.sfn = start_filter_num
        self.lada = lada
        self.slow_iter = 500
        self.input_size = input_size
        self.learning_rate = None
        self.valid_cross_entropy = tf.placeholder(tf.float32, [])
        self.valid_iou = tf.placeholder(tf.float32, [])
        self.valid_d_loss = tf.placeholder(tf.float32, [])
        self.valid_g_loss = tf.placeholder(tf.float32, [])
        self.valid_images = tf.placeholder(
            tf.uint8, shape=[None, input_size[0] - self.get_overlap(), (input_size[1] - self.get_overlap()) * 3, 3],
            name='validation_images')
        self.update_ops = None
        self.config = None
        self.hard_label = None
        self.fake_logit = None
        self.true_logit = None
        self.d_loss = None
        self.g_loss = None

    def get_overlap(self):
        return 184

    def make_encoder(self, x_name):
        sfn = self.sfn

        # downsample
        conv1, pool1 = self.conv_conv_pool(self.inputs[x_name], [sfn, sfn], self.trainable, name='conv1',
                                           padding='valid', dropout=self.dropout_rate)
        conv2, pool2 = self.conv_conv_pool(pool1, [sfn * 2, sfn * 2], self.trainable, name='conv2',
                                           padding='valid', dropout=self.dropout_rate)
        conv3, pool3 = self.conv_conv_pool(pool2, [sfn * 4, sfn * 4], self.trainable, name='conv3',
                                           padding='valid', dropout=self.dropout_rate)
        conv4, pool4 = self.conv_conv_pool(pool3, [sfn * 8, sfn * 8], self.trainable, name='conv4',
                                           padding='valid', dropout=self.dropout_rate)
        pool5 = self.conv_conv_pool(pool4, [sfn * 16, sfn * 16], self.trainable, name='conv5', pool=False,
                                    padding='valid', dropout=self.dropout_rate)

        # upsample
        up6 = self.crop_upsample_concat(pool5, conv4, 8, name='6')
        conv6 = self.conv_conv_pool(up6, [sfn * 8, sfn * 8], self.trainable, name='up6', pool=False,
                                    padding='valid', dropout=self.dropout_rate)
        up7 = self.crop_upsample_concat(conv6, conv3, 32, name='7')
        conv7 = self.conv_conv_pool(up7, [sfn * 4, sfn * 4], self.trainable, name='up7', pool=False,
                                    padding='valid', dropout=self.dropout_rate)
        up8 = self.crop_upsample_concat(conv7, conv2, 80, name='8')
        conv8 = self.conv_conv_pool(up8, [sfn * 2, sfn * 2], self.trainable, name='up8', pool=False,
                                    padding='valid', dropout=self.dropout_rate)
        up9 = self.crop_upsample_concat(conv8, conv1, 176, name='9')
        conv9 = self.conv_conv_pool(up9, [sfn, sfn], self.trainable, name='up9', pool=False,
                                    padding='valid', dropout=self.dropout_rate)
        return conv9

    def make_discriminator(self, y, sfn=4, reuse=False):
        with tf.variable_scope('discr', reuse=reuse):
            # downsample
            conv1, pool1 = self.conv_conv_pool(y, [96, 128, 128], self.trainable, name='conv1', kernal_size=(3, 3),
                                               conv_stride=(1, 1), padding='valid', dropout=self.dropout_rate,
                                               pool=True, bn=False, activation=tf.nn.relu)
            conv2, pool2 = self.conv_conv_pool(pool1, [128, 128], self.trainable, name='conv2', kernal_size=(3, 3),
                                               conv_stride=(1, 1), padding='valid', dropout=self.dropout_rate,
                                               pool=True, bn=False, activation=tf.nn.relu)
            conv3, pool3 = self.conv_conv_pool(pool2, [256], self.trainable, name='conv3', kernal_size=(3, 3),
                                               conv_stride=(1, 1), padding='valid', dropout=self.dropout_rate,
                                               pool=True, bn=False, activation=tf.nn.relu)
            conv4, pool4 = self.conv_conv_pool(pool3, [256], self.trainable, name='conv4', kernal_size=(3, 3),
                                               conv_stride=(1, 1), padding='valid', dropout=self.dropout_rate,
                                               pool=True, bn=False, activation=tf.nn.relu)
            conv5 = tf.layers.conv2d(pool4, 2, kernel_size=(3, 3), name='layerconv4')
            flat = tf.reshape(conv5, shape=[self.bs, 19*19*2])
            return self.fc_fc(flat, [1], self.trainable, name='fc_final', activation=None, dropout=False)

    def create_graph(self, names, class_num, start_filter_num=32):
        self.class_num = class_num

        conv9 = self.make_encoder(names[0])
        self.pred = tf.layers.conv2d(conv9, class_num, (1, 1), name='final', activation=None, padding='same')
        self.output = tf.nn.softmax(self.pred)

        with tf.variable_scope('Discriminator'):
            _, w, h, _ = self.inputs[names[1]].get_shape().as_list()
            true_y = tf.cast(tf.image.resize_image_with_crop_or_pad(self.inputs[names[1]], w - self.get_overlap(),
                                                                    h - self.get_overlap()), tf.float32)
            true_x = tf.cast(tf.image.resize_image_with_crop_or_pad(self.inputs[names[0]], w - self.get_overlap(),
                                                                    h - self.get_overlap()), tf.float32)
            orig_rgb = true_x * true_y
            pred_rgb = true_x * tf.expand_dims(self.output[:, :, :, 1], axis=-1)
            self.true_logit = self.make_discriminator(orig_rgb, sfn=start_filter_num//4, reuse=False)
            self.fake_logit = self.make_discriminator(pred_rgb, sfn=start_filter_num//4, reuse=True)

    def make_loss(self, y_name, loss_type='xent', **kwargs):
        with tf.variable_scope('loss'):
            pred_flat = tf.reshape(self.pred, [-1, self.class_num])
            _, w, h, _ = self.inputs[y_name].get_shape().as_list()
            y = tf.image.resize_image_with_crop_or_pad(self.inputs[y_name], w - self.get_overlap(),
                                                       h - self.get_overlap())
            y_flat = tf.reshape(tf.squeeze(y, axis=[3]), [-1, ])
            indices = tf.squeeze(tf.where(tf.less_equal(y_flat, self.class_num - 1)), 1)
            gt = tf.gather(y_flat, indices)
            prediction = tf.gather(pred_flat, indices)

            pred = tf.argmax(prediction, axis=-1, output_type=tf.int32)
            intersect = tf.cast(tf.reduce_sum(gt * pred), tf.float32)
            union = tf.cast(tf.reduce_sum(gt), tf.float32) + tf.cast(tf.reduce_sum(pred), tf.float32) \
                    - tf.cast(tf.reduce_sum(gt * pred), tf.float32)
            self.loss_iou = tf.convert_to_tensor([intersect, union])
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt))

        with tf.variable_scope('adv_loss'):
            d_loss_fake_0 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_logit,
                                                                                   labels=tf.zeros([self.bs, 1])))
            d_loss_fake_1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_logit,
                                                                                   labels=tf.ones([self.bs, 1])))
            d_loss_real_1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.true_logit,
                                                                                   labels=tf.ones([self.bs, 1])))
            self.g_loss = self.lada * d_loss_fake_1 + self.loss
            self.d_loss = d_loss_fake_0 + d_loss_real_1

    def make_optimizer(self, train_var_filter):
        with tf.control_dependencies(self.update_ops):
            if train_var_filter is None:
                t_vars = tf.trainable_variables()
                e_vars = [var for var in t_vars if 'Discriminator' not in var.name]
                d_vars = [var for var in t_vars if 'Discriminator' in var.name]
                g_optm = tf.train.AdamOptimizer(self.learning_rate[0], name='Adam_g').\
                    minimize(self.g_loss, var_list=e_vars, global_step=self.global_step)
                d_optm = tf.train.AdamOptimizer(self.learning_rate[1], name='Adam_d').\
                    minimize(self.d_loss, var_list=d_vars, global_step=None)
                self.optimizer = [g_optm, d_optm]

    @staticmethod
    def image_summary(image, truth, prediction, img_mean=np.array((0, 0, 0), dtype=np.float32)):
        truth_img = util_functions.decode_labels(truth)

        prediction = util_functions.pad_prediction(image, prediction)
        pred_labels = util_functions.get_pred_labels(prediction)
        pred_img = util_functions.decode_labels(pred_labels)

        return np.concatenate([image + img_mean, truth_img, pred_img], axis=2)

    def train(self, x_name, y_name, n_train, sess, summary_writer, n_valid=1000,
              train_reader=None, train_reader_source=None, train_reader_target=None, valid_reader=None,
              image_summary=None, verb_step=100, save_epoch=5,
              img_mean=np.array((0, 0, 0), dtype=np.float32),
              continue_dir=None, valid_iou=False):
        # define summary operations
        valid_cross_entropy_summary_op = tf.summary.scalar('xent_validation', self.valid_cross_entropy)
        valid_iou_summary_op = tf.summary.scalar('iou_validation', self.valid_iou)
        valid_d_loss_summary_op = tf.summary.scalar('d_loss_validation', self.valid_d_loss)
        valid_g_loss_summary_op = tf.summary.scalar('g_loss_validation', self.valid_g_loss)
        valid_image_summary_op = tf.summary.image('Validation_images_summary', self.valid_images,
                                                  max_outputs=10)

        if continue_dir is not None and os.path.exists(continue_dir):
            self.load(continue_dir, sess)
            gs = sess.run(self.global_step)
            start_epoch = int(np.ceil(gs/n_train*self.bs))
            start_step = gs - int(start_epoch*n_train/self.bs)
        else:
            start_epoch = 0
            start_step = 0

        cross_entropy_valid_min = np.inf
        iou_valid_max = 0
        for epoch in range(start_epoch, self.epochs):
            start_time = time.time()
            for step in range(start_step, n_train, self.bs * self.slow_iter):
                for _ in range(self.slow_iter):
                    X_batch, y_batch = train_reader.readerAction(sess)
                    _, self.global_step_value = sess.run([self.optimizer[0], self.global_step],
                                                         feed_dict={self.inputs[x_name]:X_batch,
                                                                    self.inputs[y_name]:y_batch,
                                                                    self.trainable: True})
                    X_batch, _ = train_reader_target.readerAction(sess)
                    _, y_batch = train_reader_source.readerAction(sess)
                    sess.run([self.optimizer[1]], feed_dict={self.inputs[x_name]: X_batch,
                                                             self.inputs[y_name]: y_batch,
                                                             self.trainable: True})

                    if self.global_step_value % verb_step == 0:
                        step_cross_entropy, step_summary = sess.run([self.loss, self.summary],
                                                                    feed_dict={self.inputs[x_name]: X_batch,
                                                                               self.inputs[y_name]: y_batch,
                                                                               self.trainable: False})
                        summary_writer.add_summary(step_summary, self.global_step_value)
                        print('Epoch {:d} step {:d}\tcross entropy = {:.3f}'.
                              format(epoch, self.global_step_value, step_cross_entropy))
            # validation
            cross_entropy_valid_mean = []
            d_loss_valid_mean = []
            g_loss_valid_mean = []
            iou_valid_mean = np.zeros(2)
            X_batch_val, y_batch_val, pred_valid = None, None, None
            for step in range(0, n_valid, self.bs):
                X_batch_val, y_batch_val = valid_reader.readerAction(sess)
                pred_valid, cross_entropy_valid, iou_valid = sess.run(
                    [self.pred, self.loss, self.loss_iou], feed_dict={self.inputs[x_name]: X_batch_val,
                                                                      self.inputs[y_name]: y_batch_val,
                                                                      self.trainable: False})
                _, y_batch_val_target = train_reader_source.readerAction(sess)
                d_loss_valid, g_loss_valid = sess.run([self.d_loss, self.g_loss],
                                                      feed_dict={self.inputs[x_name]: X_batch_val,
                                                                 self.inputs[y_name]: y_batch_val_target,
                                                                 self.trainable: False})
                cross_entropy_valid_mean.append(cross_entropy_valid)
                d_loss_valid_mean.append(d_loss_valid)
                g_loss_valid_mean.append(g_loss_valid)
                iou_valid_mean += iou_valid
            cross_entropy_valid_mean = np.mean(cross_entropy_valid_mean)
            d_loss_valid_mean = np.mean(d_loss_valid_mean)
            g_loss_valid_mean = np.mean(g_loss_valid_mean)
            iou_valid_mean = iou_valid_mean[0] / iou_valid_mean[1]
            duration = time.time() - start_time
            if valid_iou:
                print('Validation IoU: {:.3f}, duration: {:.3f}'.format(iou_valid_mean, duration))
            else:
                print('Val xent: {:.3f}, g_loss: {:.3f}, d_loss: {:.3f}, duration: {:.3f}'.
                      format(cross_entropy_valid_mean, d_loss_valid_mean, g_loss_valid_mean, duration))
            valid_summaries = sess.run([valid_cross_entropy_summary_op, valid_iou_summary_op,
                                        valid_d_loss_summary_op, valid_g_loss_summary_op],
                                       feed_dict={self.valid_cross_entropy: cross_entropy_valid_mean,
                                                  self.valid_iou: iou_valid_mean,
                                                  self.valid_d_loss: d_loss_valid_mean,
                                                  self.valid_g_loss: g_loss_valid_mean})
            for i in range(4):
                summary_writer.add_summary(valid_summaries[i], self.global_step_value)
            if valid_iou:
                if iou_valid_mean > iou_valid_max:
                    iou_valid_max = iou_valid_mean
                    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)
                    saver.save(sess, '{}/best_model.ckpt'.format(self.ckdir))

            else:
                if cross_entropy_valid_mean < cross_entropy_valid_min:
                    cross_entropy_valid_min = cross_entropy_valid_mean
                    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)
                    saver.save(sess, '{}/best_model.ckpt'.format(self.ckdir))

            if image_summary is not None:
                valid_image_summary = sess.run(
                    valid_image_summary_op, feed_dict={
                        self.valid_images: image_summary(X_batch_val[:, 92:-92, 92:-92, :3],
                                                         y_batch_val[:, 92:-92, 92:-92, :],
                                                         pred_valid, img_mean)})
                summary_writer.add_summary(valid_image_summary, self.global_step_value)

            if epoch % save_epoch == 0:
                saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)
                saver.save(sess, '{}/model_{}.ckpt'.format(self.ckdir, epoch), global_step=self.global_step)

    def test(self, x_name, sess, test_iterator):
        result = []
        for X_batch in test_iterator:
            pred = sess.run(self.output, feed_dict={self.inputs[x_name]: X_batch,
                                                    self.trainable: False})
            result.append(np.concatenate([1-pred, pred], axis=-1))
        result = np.vstack(result)
        return result

    def run(self, train_reader=None, train_reader_source=None, train_reader_target=None, valid_reader=None,
            test_reader=None, pretrained_model_dir=None, layers2load=None, isTrain=False,
            img_mean=np.array((0, 0, 0), dtype=np.float32), verb_step=100, save_epoch=5, gpu=None,
            tile_size=(5000, 5000), patch_size=(572, 572), truth_val=1, continue_dir=None, load_epoch_num=None,
            valid_iou=False, best_model=True):
        if gpu is not None:
            os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
        if isTrain:
            coord = tf.train.Coordinator()
            with tf.Session(config=self.config) as sess:
                # init model
                init = [tf.global_variables_initializer(), tf.local_variables_initializer()]
                sess.run(init)
                saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)
                # load model
                if pretrained_model_dir is not None:
                    if layers2load is not None:
                        self.load_weights(pretrained_model_dir, layers2load)
                    else:
                        self.load(pretrained_model_dir, sess, saver, epoch=load_epoch_num)
                threads = tf.train.start_queue_runners(coord=coord, sess=sess)
                try:
                    train_summary_writer = tf.summary.FileWriter(self.ckdir, sess.graph)
                    self.train('X', 'Y', self.n_train, sess, train_summary_writer,
                               n_valid=self.n_valid, train_reader=train_reader, valid_reader=valid_reader,
                               train_reader_source=train_reader_source, train_reader_target=train_reader_target,
                               image_summary=self.image_summary, img_mean=img_mean,
                               verb_step=verb_step, save_epoch=save_epoch, continue_dir=continue_dir,
                               valid_iou=valid_iou)
                finally:
                    coord.request_stop()
                    coord.join(threads)
                    saver.save(sess, '{}/model.ckpt'.format(self.ckdir), global_step=self.global_step)
        else:
            if self.config is None:
                self.config = tf.ConfigProto(allow_soft_placement=True)
            pad = self.get_overlap()
            with tf.Session(config=self.config) as sess:
                init = tf.global_variables_initializer()
                sess.run(init)
                self.load(pretrained_model_dir, sess, epoch=load_epoch_num, best_model=best_model)
                self.model_name = pretrained_model_dir.split('/')[-1]
                result = self.test('X', sess, test_reader)
            image_pred = uabUtilreader.un_patchify_shrink(result,
                                                          [tile_size[0] + pad, tile_size[1] + pad],
                                                          tile_size,
                                                          patch_size,
                                                          [patch_size[0] - pad, patch_size[1] - pad],
                                                          overlap=pad)
            return util_functions.get_pred_labels(image_pred) * truth_val
