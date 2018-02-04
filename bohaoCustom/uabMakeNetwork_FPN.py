import os
import numpy as np
import tensorflow as tf
import util_functions
import uabUtilreader
from bohaoCustom import uabMakeNetwork_UNet
from bohaoCustom import uabMakeNetwork as network


class FPNRes101(uabMakeNetwork_UNet.UnetModel):
    def __init__(self, inputs, trainable, input_size, model_name='', dropout_rate=0.2,
                 learn_rate=1e-4, decay_step=60, decay_rate=0.1, epochs=100,
                 batch_size=5, start_filter_num=32):
        network.Network.__init__(self, inputs, trainable, dropout_rate,
                                 learn_rate, decay_step, decay_rate, epochs, batch_size)
        self.name = 'FPNRes101'
        self.model_name = self.get_unique_name(model_name)
        self.sfn = start_filter_num
        self.learning_rate = None
        self.valid_cross_entropy = tf.placeholder(tf.float32, [])
        self.valid_images = tf.placeholder(tf.uint8, shape=[None, input_size[0],
                                                            input_size[1] * 3, 3], name='validation_images')
        self.channel_axis = 3
        self.update_ops = None
        self.encoder_name = 'res101'

    def create_graph(self, x_name, class_num):
        self.class_num = class_num
        self.input_size = self.inputs[x_name].shape[1:3]

        print("-----------build encoder: %s-----------" % self.encoder_name)
        outputs = self._start_block('conv1', x_name)
        print("after start block:", outputs.shape)
        c2, c3, c4, c5 = self.build_encoder(x_name)

        print("-----------build decoder-----------")
        with tf.variable_scope('decoder'):
            l1 = self.right_conv(c5, 256, 'right_0')
            l2 = self.down_add(l1, c4, 256, 'up1')
            print('after up 1:', l2.shape)
            l3 = self.down_add(l2, c3, 256, 'up2')
            print('after up 2:', l3.shape)
            l4 = self.down_add(l3, c2, 256, 'up3')
            print('after up 3:', l4.shape)

            l5 = self.fpn(l1, l2, l3, l4, x_name)
            print('after up 4:', l5.shape)
            self.pred = self._conv2d(l5, 3, self.class_num, 1, 'conv_final')
            print('pred shape:', self.pred.shape)

            self.output = tf.nn.softmax(self.pred)

    def build_encoder(self, x_name):
        print("-----------build encoder-----------" )
        scope_name = 'resnet_v1_101'
        with tf.variable_scope(scope_name) as scope:
            outputs = self._start_block('conv1', x_name)
            print("after start block:", outputs.shape)
            with tf.variable_scope('block1') as scope:
                outputs = self._bottleneck_resblock(outputs, 256, 'unit_1', identity_connection=False)
                outputs = self._bottleneck_resblock(outputs, 256, 'unit_2')
                c2 = self._bottleneck_resblock(outputs, 256, 'unit_3')
                print("after block1:", c2.shape)
            with tf.variable_scope('block2') as scope:
                outputs = self._bottleneck_resblock(c2, 512, 'unit_1', half_size=True, identity_connection=False)
                for i in range(2, 5):
                    outputs = self._bottleneck_resblock(outputs, 512, 'unit_%d' % i)
                c3 = outputs
                print("after block2:", c3.shape)
            with tf.variable_scope('block3') as scope:
                outputs = self._bottleneck_resblock(c3, 1024, 'unit_1', half_size=True, identity_connection=False)
                num_layers_block3 = 23
                for i in range(2, num_layers_block3 + 1):
                    outputs = self._bottleneck_resblock(outputs, 1024, 'unit_%d' % i)
                c4 = outputs
                print("after block3:", c4.shape)
            with tf.variable_scope('block4') as scope:
                outputs = self._bottleneck_resblock(c4, 2048, 'unit_1', half_size=True, identity_connection=False)
                outputs = self._bottleneck_resblock(outputs, 2048, 'unit_2')
                c5 = self._bottleneck_resblock(outputs, 2048, 'unit_3')
                print("after block4:", outputs.shape)
                return c2, c3, c4, c5

    def right_conv(self, x, c_num, name):
        return self._conv2d(x, 1, c_num, 1, name=name)

    def down_upsample(self, x, multi, name, bilinear=False):
        _, w, h, _ = x.get_shape().as_list()
        if bilinear:
            return tf.image.resize_bilinear(x, (w * multi, h * multi), name='upsample_{}'.format(name))
        else:
            return tf.image.resize_nearest_neighbor(x, (w * multi, h * multi), name='upsample_{}'.format(name))

    def down_add(self, up_x, left_x, c_num, name):
        up_x = self.down_upsample(up_x, 2, name)
        left_x = self.right_conv(left_x, c_num, name)
        return tf.add(up_x, left_x)


    #******************run_id = 1*********************#
    def class_subnet(self, x, c_num, x_name, reuse=True):
        with tf.variable_scope('classnet', reuse=reuse):
            input_size = x.shape[1:3]
            x = tf.concat([x, tf.image.resize_nearest_neighbor(self.inputs[x_name], input_size)], axis=3)
            output = self._conv2d(x, 3, c_num, 1, name='classnet_1')
            output = self._conv2d(output, 3, c_num, 1, name='classnet_2')
            return self._conv2d(output, 3, self.class_num, 1, name='classnet_3')

    def fpn(self, l1, l2, l3, l4, x_name):
        #***************run_id = 1*******************#
        l1_up = self.down_upsample(self.class_subnet(l1, 256, x_name, reuse=False), 32, 'up1', bilinear=True)
        l2_up = self.down_upsample(self.class_subnet(l2, 256, x_name), 16, 'up2', bilinear=True)
        l3_up = self.down_upsample(self.class_subnet(l3, 256, x_name), 8, 'up3', bilinear=True)
        l4_up = self.down_upsample(self.class_subnet(l4, 256, x_name), 4, 'up4', bilinear=True)
        outputs = tf.concat([l1_up, l2_up, l3_up, l4_up], axis=3)

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
            tile_size=(5000, 5000), patch_size=(572, 572), truth_val=1, continue_dir=None):
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
                        restore_var = [v for v in tf.global_variables() if 'resnet_v1' in v.name
                                       and 'Adam' not in v.name]
                        loader = tf.train.Saver(var_list=restore_var)
                        self.load(pretrained_model_dir, sess, loader)
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
                self.load(pretrained_model_dir, sess)
                self.model_name = pretrained_model_dir.split('/')[-1]
                result = self.test('X', sess, test_reader)
            image_pred = uabUtilreader.un_patchify(result, tile_size, patch_size)
            return util_functions.get_pred_labels(image_pred) * truth_val