"""
This architecture comes from this paper: https://arxiv.org/abs/1611.08323
Please refer to Figure 3 and Table 1 for details and naming rules in this file

Note: bias is missing in this implementation
"""
import tensorflow as tf
from bohaoCustom import uabMakeNetwork_UNet
from bohaoCustom import uabMakeNetwork as network


def conv_bn_relu(input_x, filter_num, training, name_id,
                 filter_size=(3, 3)):
    input_x = tf.layers.conv2d(input_x, filter_num, filter_size,
                               activation=None, padding='same', name='conv_{}'.format(name_id))
    input_x = tf.layers.batch_normalization(input_x, training=training, name='bn_{}'.format(name_id))
    input_x = tf.nn.relu(input_x, name='relu_{}'.format(name_id))
    return input_x


def r_unit(input_x, filter_num, training, name):
    with tf.variable_scope('runit{}'.format(name)):
        x_conv = conv_bn_relu(input_x, filter_num, training, name_id=1)
        x_conv = conv_bn_relu(x_conv, filter_num, training, name_id=2)
        output_x = tf.add(input_x, x_conv)
        return output_x


def frr_unit(input_y, input_z, name, reduce_num, filter_num, filter_num_out, training):
    with tf.variable_scope('frrunit{}'.format(name)):
        z_pool = tf.layers.max_pooling2d(input_z, (reduce_num, reduce_num),
                                         strides=(reduce_num, reduce_num), name='pool_{}'.format(name))
        zy_concat = tf.concat([input_y, z_pool], axis=-1, name='concat_{}'.format(name))
        zy_concat = conv_bn_relu(zy_concat, filter_num, training, name_id=1)
        zy_concat = conv_bn_relu(zy_concat, filter_num, training, name_id=2)
        zy_channel_change = tf.layers.conv2d(zy_concat, filter_num_out, (1, 1),
                                             activation=None, padding='same', name='conv_classify')

        H, W, _ = zy_channel_change.get_shape().as_list()[1:]
        target_H = H * reduce_num
        target_W = W * reduce_num
        z_out = tf.image.resize_nearest_neighbor(zy_channel_change, (target_H, target_W), name='upsample_{}'.format(name))

        z_out = tf.add(input_z, z_out)

        return zy_concat, z_out


class FRRN(uabMakeNetwork_UNet.UnetModel):
    def __init__(self, inputs, trainable, input_size, model_name='', dropout_rate=0.2,
                 learn_rate=1e-4, decay_step=60, decay_rate=0.1, epochs=100,
                 batch_size=5, start_filter_num=32):
        network.Network.__init__(self, inputs, trainable, dropout_rate,
                                 learn_rate, decay_step, decay_rate, epochs, batch_size)
        self.name = 'FRRN'
        self.model_name = self.get_unique_name(model_name)
        self.sfn = start_filter_num
        self.learning_rate = None
        self.valid_cross_entropy = tf.placeholder(tf.float32, [])
        self.valid_iou = tf.placeholder(tf.float32, [])
        self.valid_images = tf.placeholder(tf.uint8, shape=[None, input_size[0],
                                                            input_size[1] * 3, 3], name='validation_images')
        self.update_ops = None

    def create_graph(self, x_name, class_num, start_filter_num=32):
        self.class_num = class_num
        sfn = self.sfn

        # pre-pooling
        input_conv = conv_bn_relu(self.inputs[x_name], sfn, self.trainable, name_id='initial', filter_size=(5, 5))
        input_conv = r_unit(input_conv, sfn, self.trainable, name=1)
        input_conv = r_unit(input_conv, sfn, self.trainable, name=2)
        input_conv = r_unit(input_conv, sfn, self.trainable, name=3)

        # pooling
        with tf.variable_scope('conv1'):
            input_pool1 = tf.layers.max_pooling2d(input_conv, (2, 2), strides=(2, 2), name='stream_pool_1')
            residual_1 = tf.layers.conv2d(input_conv, 32, (1, 1), activation=None, padding='same',
                                          name='stream_conv_0')
            for i in range(3):
                input_pool1, residual_1 = frr_unit(input_pool1, residual_1, i+1, 2, 2*sfn, 32, self.trainable)
        with tf.variable_scope('conv2'):
            input_pool2 = tf.layers.max_pooling2d(input_pool1, (2, 2), strides=(2, 2), name='stream_pool_2')
            residual_2 = residual_1
            for i in range(4):
                input_pool2, residual_2 = frr_unit(input_pool2, residual_2, i+1, 4, 4*sfn, 32, self.trainable)
        with tf.variable_scope('conv3'):
            input_pool3 = tf.layers.max_pooling2d(input_pool2, (2, 2), strides=(2, 2), name='stream_pool_3')
            residual_3 = residual_2
            for i in range(2):
                input_pool3, residual_3 = frr_unit(input_pool3, residual_3, i+1, 8, 8*sfn, 32, self.trainable)
        with tf.variable_scope('conv4'):
            input_pool4 = tf.layers.max_pooling2d(input_pool3, (2, 2), strides=(2, 2), name='stream_pool_4')
            residual_4 = residual_3
            for i in range(2):
                input_pool4, residual_4 = frr_unit(input_pool4, residual_4, i+1, 16, 8*sfn, 32, self.trainable)

        # uppooling
        with tf.variable_scope('up5'):
            input_up5 = self.upsampling_2D(input_pool4, size=(2, 2), name='upsample_5')
            residual_5 = residual_4
            for i in range(2):
                input_up5, residual_5 = frr_unit(input_up5, residual_5, i+1, 8, 4*sfn, 32, self.trainable)
        with tf.variable_scope('up6'):
            input_up6 = self.upsampling_2D(input_up5, size=(2, 2), name='upsample_6')
            residual_6 = residual_5
            for i in range(2):
                input_up6, residual_6 = frr_unit(input_up6, residual_6, i+1, 4, 4*sfn, 32, self.trainable)
        with tf.variable_scope('up7'):
            input_up7 = self.upsampling_2D(input_up6, size=(2, 2), name='upsample_7')
            residual_7 = residual_6
            for i in range(2):
                input_up7, residual_7 = frr_unit(input_up7, residual_7, i+1, 2, 2*sfn, 32, self.trainable)

        # post pooling
        input_up8 = self.upsampling_2D(input_up7, size=(2, 2), name='upsample_8')
        input_concat = tf.concat([input_up8, residual_7], axis=-1, name='concat_postpool')
        input_concat = r_unit(input_concat, sfn*3, self.trainable, name=4)
        input_concat = r_unit(input_concat, sfn*3, self.trainable, name=5)
        input_concat = r_unit(input_concat, sfn*3, self.trainable, name=6)

        self.pred = tf.layers.conv2d(input_concat, class_num, (1, 1), name='final', activation=None, padding='same')
        self.output = tf.nn.softmax(self.pred)
