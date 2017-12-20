import tensorflow as tf
from bohaoCustom import uabMakeNetwork as network
from bohaoCustom import uabMakeNetwork_UNet


class ResFcnModel(uabMakeNetwork_UNet.UnetModel):
    def __init__(self, inputs, trainable, input_size, model_name='', dropout_rate=0.2,
                 learn_rate=1e-4, decay_step=60, decay_rate=0.1, epochs=100,
                 batch_size=5, start_filter_num=32):
        network.Network.__init__(self, inputs, trainable, dropout_rate,
                                 learn_rate, decay_step, decay_rate, epochs, batch_size)
        self.name = 'ResFcn'
        self.model_name = self.get_unique_name(model_name)
        self.sfn = start_filter_num
        self.learning_rate = None
        self.valid_cross_entropy = tf.placeholder(tf.float32, [])
        self.valid_images = tf.placeholder(tf.uint8, shape=[None, input_size[0],
                                                            input_size[1] * 3, 3], name='validation_images')
        self.update_ops = None
        self.config = None
        self.n_train = 0
        self.n_valid = 0

    def conv_block(self, input_, n_filters, training, name, kernal_size=(3, 3),
                   stride=(2,2), activation=tf.nn.relu):
        filters1, filters2, filters3 = n_filters
        with tf.variable_scope('layer{}'.format(name)):
            net = input_

            net = tf.layers.conv2d(net, filters1, (1, 1), strides=stride,
                                   padding='valid', activation=None, name='conv_1')
            net = tf.layers.batch_normalization(net, training=training, name='bn_1')
            net = activation(net, name='relu_1')

            net = tf.layers.conv2d(net, filters2, kernal_size, activation=None,
                                   padding='same', name='conv_2')
            net = tf.layers.batch_normalization(net, training=training, name='bn_2')
            net = activation(net, name='relu_2')

            net = tf.layers.conv2d(net, filters3, (1, 1), activation=None,
                                   padding='valid', name='conv_3')
            net = tf.layers.batch_normalization(net, training=training, name='bn_3')

            short_cut = tf.layers.conv2d(input_, filters3, (1, 1), strides=stride,
                                         padding='valid', activation=None, name='conv_cut')
            short_cut = tf.layers.batch_normalization(short_cut, training=training, name='bn_cut')

            net = tf.add(net, short_cut)
            net = activation(net, name='relu_3')

            return net

    def identity_block(self, input_, n_filters, training, name, kernal_size=(3, 3),
                   activation=tf.nn.relu):
        filters1, filters2, filters3 = n_filters
        with tf.variable_scope('layer{}'.format(name)):
            net = input_

            net = tf.layers.conv2d(net, filters1, (1, 1),
                                   padding='valid', activation=None, name='conv_1')
            net = tf.layers.batch_normalization(net, training=training, name='bn_1')
            net = activation(net, name='relu_1')

            net = tf.layers.conv2d(net, filters2, kernal_size, activation=None,
                                   padding='same', name='conv_2')
            net = tf.layers.batch_normalization(net, training=training, name='bn_2')
            net = activation(net, name='relu_2')

            net = tf.layers.conv2d(net, filters3, (1, 1),
                                   padding='valid', activation=None, name='conv_3')
            net = tf.layers.batch_normalization(net, training=training, name='bn_3')

            net = tf.add(net, input_)
            net = activation(net, name='relu_3')

            return net

    def create_graph(self, x_name, class_num):
        self.class_num = class_num
        sfn = self.sfn
        H, W, _ = self.inputs[x_name].get_shape().as_list()[1:]

        # pad & down sample
        with tf.variable_scope('layer_initial'):
            net = tf.pad(self.inputs[x_name], tf.constant([[0,0], [3,3], [3,3], [0,0]]))
            net = tf.layers.conv2d(net, sfn, (7, 7), strides=(2, 2), activation=None, padding='valid',
                                   name='conv_initial')
            net = tf.layers.batch_normalization(net, training=self.trainable, name='bn_initial')
            net = tf.nn.relu(net, name='relu_initial')
            net = tf.layers.max_pooling2d(net, (3, 3), strides=(2, 2), name='pool_initial')

        # blocks
        with tf.variable_scope('layer_1'):
            down1 = self.conv_block(net, [sfn, sfn, sfn*4], self.trainable, name='conv1_1', stride=(1, 1))
            down2 = self.identity_block(down1, [sfn, sfn, sfn*4], self.trainable, name='identity1_1')
            down3 = self.identity_block(down2, [sfn, sfn, sfn*4], self.trainable, name='identity1_2')

        with tf.variable_scope('layer_2'):
            down4 = self.conv_block(down3, [sfn*2, sfn*2, sfn*8], self.trainable, name='conv2_1')
            down5 = self.identity_block(down4, [sfn*2, sfn*2, sfn*8], self.trainable, name='identity2_1')
            down6 = self.identity_block(down5, [sfn*2, sfn*2, sfn*8], self.trainable, name='identity2_2')
            down7 = self.identity_block(down6, [sfn*2, sfn*2, sfn*8], self.trainable, name='identity2_3')

        with tf.variable_scope('layer_3'):
            down8 = self.conv_block(down7, [sfn*4, sfn*4, sfn*16], self.trainable, name='conv3_1')
            down9 = self.identity_block(down8, [sfn*4, sfn*4, sfn*16], self.trainable, name='identity3_1')
            down10 = self.identity_block(down9, [sfn*4, sfn*4, sfn*16], self.trainable, name='identity3_2')
            down11 = self.identity_block(down10, [sfn*4, sfn*4, sfn*16], self.trainable, name='identity3_3')
            down12 = self.identity_block(down11, [sfn*4, sfn*4, sfn*16], self.trainable, name='identity3_4')
            down13 = self.identity_block(down12, [sfn*4, sfn*4, sfn*16], self.trainable, name='identity3_5')

        with tf.variable_scope('layer_4'):
            down14 = self.conv_block(down13, [sfn*8, sfn*8, sfn*32], self.trainable, name='conv4_1')
            down15 = self.identity_block(down14, [sfn*8, sfn*8, sfn*32], self.trainable, name='identity4_1')
            down16 = self.identity_block(down15, [sfn*8, sfn*8, sfn*32], self.trainable, name='identity4_2')

        # upsample & fuse
        with tf.variable_scope('up_1'):
            up1 = tf.layers.conv2d(down7, self.class_num, (1, 1), activation=None, name='conv_1')
            up1 = tf.image.resize_bilinear(up1, tf.constant([H, W]), name='upsample_1')

        with tf.variable_scope('up_2'):
            up2 = tf.layers.conv2d(down13, self.class_num, (1, 1), activation=None, name='conv_1')
            up2 = tf.image.resize_bilinear(up2, tf.constant([H, W]), name='upsample_1')

        with tf.variable_scope('up_3'):
            up3 = tf.layers.conv2d(down16, self.class_num, (1, 1), activation=None, name='conv_1')
            up3 = tf.image.resize_bilinear(up3, tf.constant([H, W]), name='upsample_1')

        pred = tf.add(up1, up2)
        self.pred = tf.add(pred, up3)
