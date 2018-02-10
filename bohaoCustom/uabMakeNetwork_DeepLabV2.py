"""
The original paper:
https://arxiv.org/pdf/1606.00915.pdf
This architecture comes from this implementation:
https://github.com/zhengyang-wang/Deeplab-v2--ResNet-101--Tensorflow/blob/master/model.py
"""
import os
import time
import numpy as np
import tensorflow as tf
import util_functions
import uabUtilreader
from bohaoCustom import uabMakeNetwork_UNet
from bohaoCustom import uabMakeNetwork as network


class DeeplabV2(uabMakeNetwork_UNet.UnetModel):
    def __init__(self, inputs, trainable, input_size, model_name='', dropout_rate=0.2,
                 learn_rate=1e-4, decay_step=60, decay_rate=0.1, epochs=100,
                 batch_size=5, start_filter_num=32):
        network.Network.__init__(self, inputs, trainable, dropout_rate,
                                 learn_rate, decay_step, decay_rate, epochs, batch_size)
        self.name = 'DeeplabV2'
        self.model_name = self.get_unique_name(model_name)
        self.sfn = start_filter_num
        self.learning_rate = None
        self.valid_cross_entropy = tf.placeholder(tf.float32, [])
        self.valid_images = tf.placeholder(tf.uint8, shape=[None, input_size[0],
                                                            input_size[1] * 3, 3], name='validation_images')
        self.channel_axis = 3
        self.update_ops = None

    def create_graph(self, x_name, class_num):
        self.class_num = class_num
        self.input_size = self.inputs[x_name].shape[1:3]

        self.encoding = self.build_encoder(x_name)
        self.pred = self.build_decoder(self.encoding)
        self.output_size = self.pred.shape[1:3]

        self.output = tf.image.resize_bilinear(tf.nn.softmax(self.pred), self.input_size)

    def make_loss(self, y_name, loss_type='xent'):
        # TODO loss type IoU
        with tf.variable_scope('loss'):
            pred_flat = tf.reshape(self.pred, [-1, self.class_num])
            y_resize = tf.image.resize_nearest_neighbor(self.inputs[y_name], self.output_size)
            y_flat = tf.reshape(tf.squeeze(y_resize, axis=[3]), [-1, ])
            indices = tf.squeeze(tf.where(tf.less_equal(y_flat, self.class_num - 1)), 1)
            gt = tf.gather(y_flat, indices)
            prediction = tf.gather(pred_flat, indices)
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt))

    def train(self, x_name, y_name, n_train, sess, summary_writer, n_valid=1000,
              train_reader=None, valid_reader=None,
              image_summary=None, verb_step=100, save_epoch=5,
              img_mean=np.array((0, 0, 0), dtype=np.float32),
              continue_dir=None):
        # define summary operations
        valid_cross_entropy_summary_op = tf.summary.scalar('xent_validation', self.valid_cross_entropy)
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

        cross_entropy_valid_min = np.inf
        for epoch in range(start_epoch, self.epochs):
            start_time = time.time()
            for step in range(start_step, n_train, self.bs):
                X_batch, y_batch = train_reader.readerAction(sess)
                _, self.global_step_value = sess.run([self.optimizer, self.global_step],
                                                     feed_dict={self.inputs[x_name]:X_batch,
                                                                self.inputs[y_name]:y_batch,
                                                                self.trainable: True})
                if self.global_step_value % verb_step == 0:
                    pred_train, step_cross_entropy, step_summary = sess.run([self.pred, self.loss, self.summary],
                                                                            feed_dict={self.inputs[x_name]: X_batch,
                                                                                       self.inputs[y_name]: y_batch,
                                                                                       self.trainable: False})
                    summary_writer.add_summary(step_summary, self.global_step_value)
                    print('Epoch {:d} step {:d}\tcross entropy = {:.3f}'.
                          format(epoch, self.global_step_value, step_cross_entropy))
            # validation
            cross_entropy_valid_mean = []
            for step in range(0, n_valid, self.bs):
                X_batch_val, y_batch_val = valid_reader.readerAction(sess)
                pred_valid, cross_entropy_valid = sess.run([self.pred, self.loss],
                                                           feed_dict={self.inputs[x_name]: X_batch_val,
                                                                      self.inputs[y_name]: y_batch_val,
                                                                      self.trainable: False})
                cross_entropy_valid_mean.append(cross_entropy_valid)
            cross_entropy_valid_mean = np.mean(cross_entropy_valid_mean)
            duration = time.time() - start_time
            print('Validation cross entropy: {:.3f}, duration: {:.3f}'.format(cross_entropy_valid_mean, duration))
            valid_cross_entropy_summary = sess.run(valid_cross_entropy_summary_op,
                                                   feed_dict={self.valid_cross_entropy: cross_entropy_valid_mean})
            summary_writer.add_summary(valid_cross_entropy_summary, self.global_step_value)
            if cross_entropy_valid_mean < cross_entropy_valid_min:
                cross_entropy_valid_min = cross_entropy_valid_mean
                saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)
                saver.save(sess, '{}/best_model.ckpt'.format(self.ckdir), global_step=self.global_step)

            if image_summary is not None:
                pred_valid = sess.run(tf.image.resize_bilinear(pred_valid, self.input_size))
                valid_image_summary = sess.run(valid_image_summary_op,
                                               feed_dict={self.valid_images:
                                                              image_summary(X_batch_val[:,:,:,:3], y_batch_val, pred_valid,
                                                                            img_mean)})
                summary_writer.add_summary(valid_image_summary, self.global_step_value)

            if epoch % save_epoch == 0:
                saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)
                saver.save(sess, '{}/model_{}.ckpt'.format(self.ckdir, epoch), global_step=self.global_step)

    def run(self, train_reader=None, valid_reader=None, test_reader=None, pretrained_model_dir=None, layers2load=None,
            isTrain=False, img_mean=np.array((0, 0, 0), dtype=np.float32), verb_step=100, save_epoch=5, gpu=None,
            tile_size=(5000, 5000), patch_size=(572, 572), truth_val=1, continue_dir=None, load_epoch_num=None):
        if gpu is not None:
            os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
        if isTrain:
            coord = tf.train.Coordinator()
            with tf.Session(config=self.config) as sess:
                # init model
                init = tf.global_variables_initializer()
                sess.run(init)
                saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)
                # load model
                if pretrained_model_dir is not None:
                    if layers2load is not None:
                        self.load_weights(pretrained_model_dir, layers2load)
                    else:
                        restore_var = [v for v in tf.global_variables() if 'fc' not in v.name]
                        loader = tf.train.Saver(var_list=restore_var)
                        self.load(pretrained_model_dir, sess, loader, epoch=load_epoch_num)
                threads = tf.train.start_queue_runners(coord=coord, sess=sess)
                try:
                    train_summary_writer = tf.summary.FileWriter(self.ckdir, sess.graph)
                    self.train('X', 'Y', self.n_train, sess, train_summary_writer,
                               n_valid=self.n_valid, train_reader=train_reader, valid_reader=valid_reader,
                               image_summary=util_functions.image_summary, img_mean=img_mean,
                               verb_step=verb_step, save_epoch=save_epoch, continue_dir=continue_dir)
                finally:
                    coord.request_stop()
                    coord.join(threads)
                    saver.save(sess, '{}/model.ckpt'.format(self.ckdir), global_step=self.global_step)
        else:
            with tf.Session() as sess:
                init = tf.global_variables_initializer()
                sess.run(init)
                self.load(pretrained_model_dir, sess, epoch=load_epoch_num)
                self.model_name = pretrained_model_dir.split('/')[-1]
                result = self.test('X', sess, test_reader)
            image_pred = uabUtilreader.un_patchify(result, tile_size, patch_size)
            return util_functions.get_pred_labels(image_pred) * truth_val

    def build_encoder(self, x_name):
        print("-----------build encoder: deeplab initial-----------")
        outputs = self._start_block(x_name)
        print("after start block:", outputs.shape)
        outputs = self._bottleneck_resblock(outputs, 256, '2a', identity_connection=False)
        outputs = self._bottleneck_resblock(outputs, 256, '2b')
        outputs = self._bottleneck_resblock(outputs, 256, '2c')
        print("after block1:", outputs.shape)
        outputs = self._bottleneck_resblock(outputs, 512, '3a', half_size=True, identity_connection=False)
        for i in range(1, 4):
            outputs = self._bottleneck_resblock(outputs, 512, '3b%d' % i)
        print("after block2:", outputs.shape)
        outputs = self._dilated_bottle_resblock(outputs, 1024, 2, '4a', identity_connection=False)
        for i in range(1, 23):
            outputs = self._dilated_bottle_resblock(outputs, 1024, 2, '4b%d' % i)
        print("after block3:", outputs.shape)
        outputs = self._dilated_bottle_resblock(outputs, 2048, 4, '5a', identity_connection=False)
        outputs = self._dilated_bottle_resblock(outputs, 2048, 4, '5b')
        outputs = self._dilated_bottle_resblock(outputs, 2048, 4, '5c')
        print("after block4:", outputs.shape)
        return outputs

    def build_decoder(self, encoding):
        print("-----------build decoder-----------")
        outputs = self._ASPP(encoding, self.class_num, [6, 12, 18, 24])
        #outputs = tf.image.resize_bilinear(outputs, self.input_size)
        print("after aspp block:", outputs.shape)
        return outputs

    # blocks
    def _start_block(self, x_name):
        outputs = self._conv2d(self.inputs[x_name], 7, 64, 2, name='conv1')
        outputs = self._batch_norm(outputs, name='bn_conv1', is_training=False, activation_fn=tf.nn.relu)
        outputs = self._max_pool2d(outputs, 3, 2, name='pool1')
        return outputs

    def _bottleneck_resblock(self, x, num_o, name, half_size=False, identity_connection=True):
        first_s = 2 if half_size else 1
        assert num_o % 4 == 0, 'Bottleneck number of output ERROR!'
        # branch1
        if not identity_connection:
            o_b1 = self._conv2d(x, 1, num_o, first_s, name='res%s_branch1' % name)
            o_b1 = self._batch_norm(o_b1, name='bn%s_branch1' % name, is_training=False, activation_fn=None)
        else:
            o_b1 = x
        # branch2
        o_b2a = self._conv2d(x, 1, num_o / 4, first_s, name='res%s_branch2a' % name)
        o_b2a = self._batch_norm(o_b2a, name='bn%s_branch2a' % name, is_training=False, activation_fn=tf.nn.relu)

        o_b2b = self._conv2d(o_b2a, 3, num_o / 4, 1, name='res%s_branch2b' % name)
        o_b2b = self._batch_norm(o_b2b, name='bn%s_branch2b' % name, is_training=False, activation_fn=tf.nn.relu)

        o_b2c = self._conv2d(o_b2b, 1, num_o, 1, name='res%s_branch2c' % name)
        o_b2c = self._batch_norm(o_b2c, name='bn%s_branch2c' % name, is_training=False, activation_fn=None)
        # add
        outputs = self._add([o_b1, o_b2c], name='res%s' % name)
        # relu
        outputs = self._relu(outputs, name='res%s_relu' % name)
        return outputs

    def _dilated_bottle_resblock(self, x, num_o, dilation_factor, name, identity_connection=True):
        assert num_o % 4 == 0, 'Bottleneck number of output ERROR!'
        # branch1
        if not identity_connection:
            o_b1 = self._conv2d(x, 1, num_o, 1, name='res%s_branch1' % name)
            o_b1 = self._batch_norm(o_b1, name='bn%s_branch1' % name, is_training=False, activation_fn=None)
        else:
            o_b1 = x
        # branch2
        o_b2a = self._conv2d(x, 1, num_o / 4, 1, name='res%s_branch2a' % name)
        o_b2a = self._batch_norm(o_b2a, name='bn%s_branch2a' % name, is_training=False, activation_fn=tf.nn.relu)

        o_b2b = self._dilated_conv2d(o_b2a, 3, num_o / 4, dilation_factor, name='res%s_branch2b' % name)
        o_b2b = self._batch_norm(o_b2b, name='bn%s_branch2b' % name, is_training=False, activation_fn=tf.nn.relu)

        o_b2c = self._conv2d(o_b2b, 1, num_o, 1, name='res%s_branch2c' % name)
        o_b2c = self._batch_norm(o_b2c, name='bn%s_branch2c' % name, is_training=False, activation_fn=None)
        # add
        outputs = self._add([o_b1, o_b2c], name='res%s' % name)
        # relu
        outputs = self._relu(outputs, name='res%s_relu' % name)
        return outputs

    def _ASPP(self, x, num_o, dilations):
        o = []
        for i, d in enumerate(dilations):
            o.append(self._dilated_conv2d(x, 3, num_o, d, name='fc1_voc12_c%d' % i, biased=True))
        return self._add(o, name='fc1_voc12')

        # layers

    def _conv2d(self, x, kernel_size, num_o, stride, name, biased=False):
        """
        Conv2d without BN or relu.
        """
        num_x = x.shape[self.channel_axis].value
        with tf.variable_scope(name) as scope:
            w = tf.get_variable('weights', shape=[kernel_size, kernel_size, num_x, num_o])
            s = [1, stride, stride, 1]
            o = tf.nn.conv2d(x, w, s, padding='SAME')
            if biased:
                b = tf.get_variable('biases', shape=[num_o])
                o = tf.nn.bias_add(o, b)
            return o

    def _dilated_conv2d(self, x, kernel_size, num_o, dilation_factor, name, biased=False):
        """
        Dilated conv2d without BN or relu.
        """
        num_x = x.shape[self.channel_axis].value
        with tf.variable_scope(name) as scope:
            w = tf.get_variable('weights', shape=[kernel_size, kernel_size, num_x, num_o])
            o = tf.nn.atrous_conv2d(x, w, dilation_factor, padding='SAME')
            if biased:
                b = tf.get_variable('biases', shape=[num_o])
                o = tf.nn.bias_add(o, b)
            return o

    def _relu(self, x, name):
        return tf.nn.relu(x, name=name)

    def _add(self, x_l, name):
        return tf.add_n(x_l, name=name)

    def _max_pool2d(self, x, kernel_size, stride, name):
        k = [1, kernel_size, kernel_size, 1]
        s = [1, stride, stride, 1]
        return tf.nn.max_pool(x, k, s, padding='SAME', name=name)

    def _batch_norm(self, x, name, is_training, activation_fn, trainable=False):
        # For a small batch size, it is better to keep
        # the statistics of the BN layers (running means and variances) frozen,
        # and to not update the values provided by the pre-trained model by setting is_training=False.
        # Note that is_training=False still updates BN parameters gamma (scale) and beta (offset)
        # if they are presented in var_list of the optimiser definition.
        # Set trainable = False to remove them from trainable_variables.
        with tf.variable_scope(name) as scope:
            o = tf.contrib.layers.batch_norm(
                x,
                scale=True,
                activation_fn=activation_fn,
                is_training=is_training,
                trainable=trainable,
                scope=scope)
            return o


class DeeplabV3(DeeplabV2):
    def __init__(self, inputs, trainable, input_size, model_name='', dropout_rate=0.2,
                 learn_rate=1e-4, decay_step=60, decay_rate=0.1, epochs=100,
                 batch_size=5, start_filter_num=32):
        network.Network.__init__(self, inputs, trainable, dropout_rate,
                                 learn_rate, decay_step, decay_rate, epochs, batch_size)
        self.name = 'DeeplabV3'
        self.model_name = self.get_unique_name(model_name)
        self.sfn = start_filter_num
        self.learning_rate = None
        self.valid_cross_entropy = tf.placeholder(tf.float32, [])
        self.valid_images = tf.placeholder(tf.uint8, shape=[None, input_size[0],
                                                            input_size[1] * 3, 3], name='validation_images')
        self.channel_axis = 3
        self.update_ops = None

    def create_graph(self, x_name, class_num):
        self.class_num = class_num
        self.input_size = self.inputs[x_name].shape[1:3]

        self.encoding = self.build_encoder(x_name)
        self.pred = self.build_decoder(self.encoding)
        self.output_size = self.pred.shape[1:3]

        self.output = tf.image.resize_bilinear(tf.nn.softmax(self.pred), self.input_size)

    def build_encoder(self, x_name):
        print("-----------build encoder-----------" )
        scope_name = 'resnet_v1_101'
        with tf.variable_scope(scope_name) as scope:
            outputs = self._start_block('conv1', x_name)
            print("after start block:", outputs.shape)
            with tf.variable_scope('block1') as scope:
                outputs = self._bottleneck_resblock(outputs, 256, 'unit_1', identity_connection=False)
                outputs = self._bottleneck_resblock(outputs, 256, 'unit_2')
                outputs = self._bottleneck_resblock(outputs, 256, 'unit_3')
                print("after block1:", outputs.shape)
            with tf.variable_scope('block2') as scope:
                outputs = self._bottleneck_resblock(outputs, 512, 'unit_1', half_size=True, identity_connection=False)
                for i in range(2, 5):
                    outputs = self._bottleneck_resblock(outputs, 512, 'unit_%d' % i)
                print("after block2:", outputs.shape)
            with tf.variable_scope('block3') as scope:
                outputs = self._dilated_bottle_resblock(outputs, 1024, 2, 'unit_1', identity_connection=False)
                num_layers_block3 = 23
                for i in range(2, num_layers_block3 + 1):
                    outputs = self._dilated_bottle_resblock(outputs, 1024, 2, 'unit_%d' % i)
                print("after block3:", outputs.shape)
            with tf.variable_scope('block4') as scope:
                outputs = self._dilated_bottle_resblock(outputs, 2048, 4, 'unit_1', identity_connection=False)
                outputs = self._dilated_bottle_resblock(outputs, 2048, 4, 'unit_2')
                outputs = self._dilated_bottle_resblock(outputs, 2048, 4, 'unit_3')
                print("after block4:", outputs.shape)
                return outputs

    def build_decoder(self, encoding):
        print("-----------build decoder-----------")
        with tf.variable_scope('decoder') as scope:
            outputs = self._ASPP(encoding, self.class_num, [6, 12, 18, 24])
            print("after aspp block:", outputs.shape)
            return outputs

            # blocks

    def _start_block(self, name, x_name):
        outputs = self._conv2d(self.inputs[x_name], 7, 64, 2, name=name)
        outputs = self._batch_norm(outputs, name=name, is_training=False, activation_fn=tf.nn.relu)
        outputs = self._max_pool2d(outputs, 3, 2, name='pool1')
        return outputs

    def _bottleneck_resblock(self, x, num_o, name, half_size=False, identity_connection=True):
        first_s = 2 if half_size else 1
        assert num_o % 4 == 0, 'Bottleneck number of output ERROR!'
        # branch1
        if not identity_connection:
            o_b1 = self._conv2d(x, 1, num_o, first_s, name='%s/bottleneck_v1/shortcut' % name)
            o_b1 = self._batch_norm(o_b1, name='%s/bottleneck_v1/shortcut' % name, is_training=False,
                                    activation_fn=None)
        else:
            o_b1 = x
        # branch2
        o_b2a = self._conv2d(x, 1, num_o / 4, first_s, name='%s/bottleneck_v1/conv1' % name)
        o_b2a = self._batch_norm(o_b2a, name='%s/bottleneck_v1/conv1' % name, is_training=False,
                                 activation_fn=tf.nn.relu)

        o_b2b = self._conv2d(o_b2a, 3, num_o / 4, 1, name='%s/bottleneck_v1/conv2' % name)
        o_b2b = self._batch_norm(o_b2b, name='%s/bottleneck_v1/conv2' % name, is_training=False,
                                 activation_fn=tf.nn.relu)

        o_b2c = self._conv2d(o_b2b, 1, num_o, 1, name='%s/bottleneck_v1/conv3' % name)
        o_b2c = self._batch_norm(o_b2c, name='%s/bottleneck_v1/conv3' % name, is_training=False, activation_fn=None)
        # add
        outputs = self._add([o_b1, o_b2c], name='%s/bottleneck_v1/add' % name)
        # relu
        outputs = self._relu(outputs, name='%s/bottleneck_v1/relu' % name)
        return outputs

    def _dilated_bottle_resblock(self, x, num_o, dilation_factor, name, identity_connection=True):
        assert num_o % 4 == 0, 'Bottleneck number of output ERROR!'
        # branch1
        if not identity_connection:
            o_b1 = self._conv2d(x, 1, num_o, 1, name='%s/bottleneck_v1/shortcut' % name)
            o_b1 = self._batch_norm(o_b1, name='%s/bottleneck_v1/shortcut' % name, is_training=False,
                                    activation_fn=None)
        else:
            o_b1 = x
        # branch2
        o_b2a = self._conv2d(x, 1, num_o / 4, 1, name='%s/bottleneck_v1/conv1' % name)
        o_b2a = self._batch_norm(o_b2a, name='%s/bottleneck_v1/conv1' % name, is_training=False,
                                 activation_fn=tf.nn.relu)

        o_b2b = self._dilated_conv2d(o_b2a, 3, num_o / 4, dilation_factor, name='%s/bottleneck_v1/conv2' % name)
        o_b2b = self._batch_norm(o_b2b, name='%s/bottleneck_v1/conv2' % name, is_training=False,
                                 activation_fn=tf.nn.relu)

        o_b2c = self._conv2d(o_b2b, 1, num_o, 1, name='%s/bottleneck_v1/conv3' % name)
        o_b2c = self._batch_norm(o_b2c, name='%s/bottleneck_v1/conv3' % name, is_training=False, activation_fn=None)
        # add
        outputs = self._add([o_b1, o_b2c], name='%s/bottleneck_v1/add' % name)
        # relu
        outputs = self._relu(outputs, name='%s/bottleneck_v1/relu' % name)
        return outputs

    def _ASPP(self, x, num_o, dilations):
        o = []
        for i, d in enumerate(dilations):
            o.append(self._dilated_conv2d(x, 3, num_o, d, name='aspp/conv%d' % (i + 1), biased=True))
        return self._add(o, name='aspp/add')

        # layers

    def _conv2d(self, x, kernel_size, num_o, stride, name, biased=False):
        """
        Conv2d without BN or relu.
        """
        num_x = x.shape[self.channel_axis].value
        with tf.variable_scope(name) as scope:
            w = tf.get_variable('weights', shape=[kernel_size, kernel_size, num_x, num_o])
            s = [1, stride, stride, 1]
            o = tf.nn.conv2d(x, w, s, padding='SAME')
            if biased:
                b = tf.get_variable('biases', shape=[num_o])
                o = tf.nn.bias_add(o, b)
            return o

    def _dilated_conv2d(self, x, kernel_size, num_o, dilation_factor, name, biased=False):
        """
        Dilated conv2d without BN or relu.
        """
        num_x = x.shape[self.channel_axis].value
        with tf.variable_scope(name) as scope:
            w = tf.get_variable('weights', shape=[kernel_size, kernel_size, num_x, num_o])
            o = tf.nn.atrous_conv2d(x, w, dilation_factor, padding='SAME')
            if biased:
                b = tf.get_variable('biases', shape=[num_o])
                o = tf.nn.bias_add(o, b)
            return o

    def _relu(self, x, name):
        return tf.nn.relu(x, name=name)

    def _add(self, x_l, name):
        return tf.add_n(x_l, name=name)

    def _max_pool2d(self, x, kernel_size, stride, name):
        k = [1, kernel_size, kernel_size, 1]
        s = [1, stride, stride, 1]
        return tf.nn.max_pool(x, k, s, padding='SAME', name=name)

    def _batch_norm(self, x, name, is_training, activation_fn, trainable=False):
        # For a small batch size, it is better to keep
        # the statistics of the BN layers (running means and variances) frozen,
        # and to not update the values provided by the pre-trained model by setting is_training=False.
        # Note that is_training=False still updates BN parameters gamma (scale) and beta (offset)
        # if they are presented in var_list of the optimiser definition.
        # Set trainable = False to remove them from trainable_variables.
        with tf.variable_scope(name + '/BatchNorm') as scope:
            o = tf.contrib.layers.batch_norm(
                x,
                scale=True,
                activation_fn=activation_fn,
                is_training=is_training,
                trainable=trainable,
                scope=scope)
            return o

    def run(self, train_reader=None, valid_reader=None, test_reader=None, pretrained_model_dir=None, layers2load=None,
            isTrain=False, img_mean=np.array((0, 0, 0), dtype=np.float32), verb_step=100, save_epoch=5, gpu=None,
            tile_size=(5000, 5000), patch_size=(572, 572), truth_val=1, continue_dir=None, load_epoch_num=None):
        if gpu is not None:
            os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
        if isTrain:
            coord = tf.train.Coordinator()
            with tf.Session(config=self.config) as sess:
                # init model
                init = tf.global_variables_initializer()
                sess.run(init)
                saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)
                # load model
                if pretrained_model_dir is not None:
                    if layers2load is not None:
                        self.load_weights(pretrained_model_dir, layers2load)
                    else:
                        restore_var = [v for v in tf.global_variables() if 'resnet_v1' in v.name and
                                       not 'Adam' in v.name]
                        loader = tf.train.Saver(var_list=restore_var)
                        self.load(pretrained_model_dir, sess, loader, epoch=load_epoch_num)
                threads = tf.train.start_queue_runners(coord=coord, sess=sess)
                try:
                    train_summary_writer = tf.summary.FileWriter(self.ckdir, sess.graph)
                    self.train('X', 'Y', self.n_train, sess, train_summary_writer,
                               n_valid=self.n_valid, train_reader=train_reader, valid_reader=valid_reader,
                               image_summary=util_functions.image_summary, img_mean=img_mean,
                               verb_step=verb_step, save_epoch=save_epoch, continue_dir=continue_dir)
                finally:
                    coord.request_stop()
                    coord.join(threads)
                    saver.save(sess, '{}/model.ckpt'.format(self.ckdir), global_step=self.global_step)
        else:
            with tf.Session() as sess:
                init = tf.global_variables_initializer()
                sess.run(init)
                self.load(pretrained_model_dir, sess, epoch=load_epoch_num)
                self.model_name = pretrained_model_dir.split('/')[-1]
                result = self.test('X', sess, test_reader)
            image_pred = uabUtilreader.un_patchify(result, tile_size, patch_size)
            return util_functions.get_pred_labels(image_pred) * truth_val
