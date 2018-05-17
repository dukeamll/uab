import os
import re
import time
import imageio
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import array_ops
import util_functions
import uabUtilreader
import uabDataReader
import uabRepoPaths
from bohaoCustom import uabMakeNetwork_UNet
from bohaoCustom import uabMakeNetwork as network

class UnetModelMTL(uabMakeNetwork_UNet.UnetModelCrop):
    def __init__(self, inputs, trainable, input_size, model_name='', dropout_rate=None,
                 learn_rate=1e-4, decay_step=60, decay_rate=0.1, epochs=100,
                 batch_size=5, start_filter_num=32, source_num=2, source_name=None, source_control=None):
        network.Network.__init__(self, inputs, trainable, dropout_rate,
                                 learn_rate, decay_step, decay_rate, epochs, batch_size)
        self.name = 'UnetMTL'
        self.model_name = self.get_unique_name(model_name)
        self.sfn = start_filter_num
        self.learning_rate = None
        self.source_num = source_num
        if source_name is None or len(source_name) != source_num:
            self.source_name = ['D{}'.format(i) for i in range(self.source_num)]
        else:
            self.source_name = source_name
        if source_control is None or len(source_control) != source_num:
            self.source_control = [1 for i in range(self.source_num)]
        else:
            self.source_control = source_control
        self.valid_cross_entropy = [tf.placeholder(tf.float32, []) for i in range(self.source_num)]
        self.valid_iou = [tf.placeholder(tf.float32, []) for i in range(self.source_num)]
        self.valid_images = [tf.placeholder(tf.uint8, shape=[None, input_size[0],
                                                            input_size[1] * 3, 3], name='validation_images')
                             for i in range(self.source_num)]
        self.update_ops = None

    def create_graph(self, x_name, class_num, start_filter_num=32):
        assert len(class_num) == self.source_num
        self.class_num = class_num
        sfn = self.sfn

        # downsample
        with tf.variable_scope('Encoder'):
            conv1, pool1 = self.conv_conv_pool(self.inputs[x_name], [sfn, sfn], self.trainable, name='conv1',
                                               padding='valid', dropout=self.dropout_rate)
            conv2, pool2 = self.conv_conv_pool(pool1, [sfn*2, sfn*2], self.trainable, name='conv2',
                                               padding='valid', dropout=self.dropout_rate)
            conv3, pool3 = self.conv_conv_pool(pool2, [sfn*4, sfn*4], self.trainable, name='conv3',
                                               padding='valid', dropout=self.dropout_rate)
            conv4, pool4 = self.conv_conv_pool(pool3, [sfn*8, sfn*8], self.trainable, name='conv4',
                                               padding='valid', dropout=self.dropout_rate)
            self.encoding = self.conv_conv_pool(pool4, [sfn*16, sfn*16], self.trainable, name='conv5', pool=False,
                                        padding='valid', dropout=self.dropout_rate)

        # upsample
        self.pred = []
        self.output = []
        for cnt, s_name in enumerate(self.source_name):
            with tf.variable_scope('Decoder_{}'.format(s_name)):
                up6 = self.crop_upsample_concat(self.encoding, conv4, 8, name='6')
                conv6 = self.conv_conv_pool(up6, [sfn*8, sfn*8], self.trainable, name='up6', pool=False,
                                            padding='valid', dropout=self.dropout_rate)
                up7 = self.crop_upsample_concat(conv6, conv3, 32, name='7')
                conv7 = self.conv_conv_pool(up7, [sfn*4, sfn*4], self.trainable, name='up7', pool=False,
                                            padding='valid', dropout=self.dropout_rate)
                up8 = self.crop_upsample_concat(conv7, conv2, 80, name='8')
                conv8 = self.conv_conv_pool(up8, [sfn*2, sfn*2], self.trainable, name='up8', pool=False,
                                            padding='valid', dropout=self.dropout_rate)
                up9 = self.crop_upsample_concat(conv8, conv1, 176, name='9')
                conv9 = self.conv_conv_pool(up9, [sfn, sfn], self.trainable, name='up9', pool=False,
                                            padding='valid', dropout=self.dropout_rate)

                self.pred.append(tf.layers.conv2d(conv9, class_num[cnt], (1, 1), name='final', activation=None,
                                                  padding='same'))
                self.output.append(tf.nn.softmax(self.pred))

    def make_loss(self, y_name, loss_type='xent', **kwargs):
        with tf.variable_scope('loss'):
            self.loss_iou = []
            self.loss = []
            for cnt in range(self.source_num):
                pred_flat = tf.reshape(self.pred[cnt], [-1, self.class_num[cnt]])
                _, w, h, _ = self.inputs[y_name].get_shape().as_list()
                y = tf.image.resize_image_with_crop_or_pad(self.inputs[y_name], w-self.get_overlap(), h-self.get_overlap())
                y_flat = tf.reshape(tf.squeeze(y, axis=[3]), [-1, ])
                indices = tf.squeeze(tf.where(tf.less_equal(y_flat, self.class_num[cnt] - 1)), 1)
                gt = tf.gather(y_flat, indices)
                prediction = tf.gather(pred_flat, indices)

                pred = tf.argmax(prediction, axis=-1, output_type=tf.int32)
                intersect = tf.cast(tf.reduce_sum(gt * pred), tf.float32)
                union = tf.cast(tf.reduce_sum(gt), tf.float32) + tf.cast(tf.reduce_sum(pred), tf.float32) \
                        - tf.cast(tf.reduce_sum(gt * pred), tf.float32)
                self.loss_iou.append(tf.convert_to_tensor([intersect, union]))
                self.loss.append(tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=prediction, labels=gt)))

    def get_overlap(self):
        # TODO calculate the padding pixels
        return 184

    def make_optimizer(self, train_var_filter=None):
        with tf.control_dependencies(self.update_ops):
            t_vars = tf.trainable_variables()
            for cnt, s_name in enumerate(self.source_name):
                var_list = [var for var in t_vars if 'Encoder' in var.name] + \
                           [var for var in t_vars if 'Decoder_{}'.format(s_name) in var.name]
                self.optimizer.append(tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss[cnt],
                                                                                          global_step=self.global_step,
                                                                                          var_list=var_list))

    def make_update_ops(self, x_name, y_name):
        tf.add_to_collection('inputs', self.inputs[x_name])
        tf.add_to_collection('inputs', self.inputs[y_name])
        for cnt in range(self.source_num):
            tf.add_to_collection('outputs', self.pred[cnt])
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    def make_summary(self, hist=False):
        if hist:
            tf.summary.histogram('Predicted Prob', tf.argmax(tf.nn.softmax(self.pred), 1))
        for cnt, s in enumerate(self.source_name):
            tf.summary.scalar('Cross Entropy {}'.format(s), self.loss[cnt])
            tf.summary.scalar('learning rate {}'.format(s), self.learning_rate)
        self.summary = tf.summary.merge_all()

    def train(self, x_name, y_name, n_train, sess, summary_writer, n_valid=1000,
              train_reader=None, valid_reader=None,
              image_summary=None, verb_step=100, save_epoch=5,
              img_mean=np.array((0, 0, 0), dtype=np.float32),
              continue_dir=None, valid_iou=False):
        # assert readers are correct
        assert len(train_reader) == self.source_num
        assert len(valid_reader) == self.source_num
        assert len(img_mean) == self.source_num

        # define summary operations
        valid_cross_entropy_summary_op = [tf.summary.scalar('xent_validation_{}'.format(self.source_name[i]),
                                                            self.valid_cross_entropy[i]) for i in range(self.source_num)]
        valid_iou_summary_op = [tf.summary.scalar('iou_validation_{}'.format(self.source_name[i]),
                                                  self.valid_iou[i]) for i in range(self.source_num)]
        valid_image_summary_op = [tf.summary.image('Validation_images_summary_{}'.format(self.source_name[i]),
                                                   self.valid_images[i], max_outputs=10) for i in range(self.source_num)]

        if continue_dir is not None and os.path.exists(continue_dir):
            self.load(continue_dir, sess)
            gs = sess.run(self.global_step)
            start_epoch = int(np.ceil(gs/n_train*self.bs))
            start_step = gs - int(start_epoch*n_train/self.bs)
        else:
            start_epoch = 0
            start_step = 0

        for epoch in range(start_epoch, self.epochs):
            start_time = time.time()
            for step in range(start_step, n_train, self.bs):
                for s in range(self.source_num):
                    for train_iter in range(self.source_control[s]):
                        X_batch, y_batch = train_reader[s].readerAction(sess)
                        _, self.global_step_value = sess.run([self.optimizer[s], self.global_step],
                                                             feed_dict={self.inputs[x_name]:X_batch,
                                                                        self.inputs[y_name]:y_batch,
                                                                        self.trainable: True})
                        if step % verb_step == 0:
                            pred_train, step_cross_entropy, step_summary = sess.run(
                                [self.pred[s], self.loss[s], self.summary],
                                feed_dict={self.inputs[x_name]: X_batch, self.inputs[y_name]: y_batch,
                                           self.trainable: False})
                            summary_writer.add_summary(step_summary, epoch*n_train+step)
                            print('Source: {} Epoch {:d} step {:d}\tcross entropy = {:.3f}'.
                                  format(self.source_name[s], epoch, step, step_cross_entropy))
            # validation
            for s in range(self.source_num):
                cross_entropy_valid_mean = []
                iou_valid_mean = np.zeros(2)
                for step in range(0, n_valid, self.bs):
                    X_batch_val, y_batch_val = valid_reader[s].readerAction(sess)
                    pred_valid, cross_entropy_valid, iou_valid = \
                        sess.run([self.pred[s], self.loss[s], self.loss_iou[s]],
                                 feed_dict={self.inputs[x_name]: X_batch_val, self.inputs[y_name]: y_batch_val,
                                            self.trainable: False})
                    cross_entropy_valid_mean.append(cross_entropy_valid)
                    iou_valid_mean += iou_valid
                cross_entropy_valid_mean = np.mean(cross_entropy_valid_mean)
                iou_valid_mean = iou_valid_mean[0] / iou_valid_mean[1]
                duration = time.time() - start_time
                if valid_iou:
                    print('Source {} Validation IoU: {:.3f}, duration: {:.3f}'.
                          format(self.source_name[s], iou_valid_mean, duration))
                else:
                    print('Source {} Validation cross entropy: {:.3f}, duration: {:.3f}'.
                          format(self.source_name[s], cross_entropy_valid_mean, duration))
                valid_cross_entropy_summary = sess.run(valid_cross_entropy_summary_op[s],
                                                       feed_dict={self.valid_cross_entropy[s]: cross_entropy_valid_mean})
                valid_iou_summary = sess.run(valid_iou_summary_op[s],
                                             feed_dict={self.valid_iou[s]: iou_valid_mean})
                summary_writer.add_summary(valid_cross_entropy_summary, epoch)
                summary_writer.add_summary(valid_iou_summary, epoch)

                if image_summary is not None:
                    valid_image_summary = sess.run(valid_image_summary_op[s],
                                                   feed_dict={self.valid_images[s]:
                                                                  image_summary(X_batch_val[:,:,:,:3], y_batch_val, pred_valid,
                                                                                img_mean[s])})
                    summary_writer.add_summary(valid_image_summary, epoch)

                if epoch % save_epoch == 0:
                    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)
                    saver.save(sess, '{}/model_{}.ckpt'.format(self.ckdir, epoch), global_step=self.global_step)
