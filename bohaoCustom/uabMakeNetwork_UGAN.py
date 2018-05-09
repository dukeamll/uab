import os
import time
import imageio
import numpy as np
import scipy.misc
import tensorflow as tf
import utils
from bohaoCustom import uabMakeNetwork as network
from bohaoCustom import uabMakeNetwork_DeepLabV2
from bohaoCustom import uabMakeNetwork_UNetEncoder


def conv_conv_shrink(input_, n_filters, training, name, kernal_size=(3, 3),
                   activation=tf.nn.relu, padding='same', bn=True):
    net = input_

    with tf.variable_scope('layer{}'.format(name)):
        for i, F in enumerate(n_filters[:-1]):
            net = tf.layers.conv2d(net, F, kernal_size, activation=None,
                                   padding=padding, name='conv_{}'.format(i + 1))
            if bn:
                net = tf.layers.batch_normalization(net, training=training, name='bn_{}'.format(i + 1))
            net = activation(net, name='relu_{}'.format(name, i + 1))

        net = tf.layers.conv2d(net, n_filters[-1], kernal_size, strides=2, activation=None,
                                   padding=padding, name='conv_shrink')
        return net


class UGAN(uabMakeNetwork_DeepLabV2.DeeplabV3):
    def __init__(self, inputs, trainable, input_size, model_name='', dropout_rate=None,
                 learn_rate=1e-4, decay_step=60, decay_rate=0.1, epochs=100,
                 batch_size=5, start_filter_num=32, z_dim=100):
        network.Network.__init__(self, inputs, trainable, dropout_rate,
                                 learn_rate, decay_step, decay_rate, epochs, batch_size)
        self.name = 'UGAN'
        self.model_name = self.get_unique_name(model_name)
        self.sfn = start_filter_num
        self.learning_rate = None
        self.valid_cross_entropy = tf.placeholder(tf.float32, [])
        self.valid_iou = tf.placeholder(tf.float32, [])
        self.valid_images = tf.placeholder(tf.uint8, shape=[None, input_size[0],
                                                            input_size[1], 3], name='validation_images')
        self.channel_axis = 3
        self.update_ops = None
        self.z_dim = z_dim

    def create_graph(self, x_name, class_num):
        self.class_num = class_num
        self.input_size = self.inputs[x_name].shape[1:3]

        self.gener = self.build_decoder(tf.reshape(self.inputs['Z'], [-1, 1, 1, self.z_dim]))

        self.discr_f, _ = self.build_encoder(self.gener, None)
        self.discr_r, _ = self.build_encoder(self.inputs[x_name], True)

    def build_encoder(self, x, reuse):
        print("-----------build encoder-----------")
        print('input:', x.shape)
        scope_name = 'encoder'
        with tf.variable_scope(scope_name, reuse=reuse) as scope:
            outputs = conv_conv_shrink(x, [32], self.trainable, name='conv1')
            outputs = conv_conv_shrink(outputs, [64], self.trainable, name='conv2')
            outputs = conv_conv_shrink(outputs, [128], self.trainable, name='conv3')
            outputs = conv_conv_shrink(outputs, [256], self.trainable, name='conv4')
            outputs = conv_conv_shrink(outputs, [512], self.trainable, name='conv5')
            outputs = conv_conv_shrink(outputs, [1024], self.trainable, name='conv6')
            outputs = conv_conv_shrink(outputs, [2048], self.trainable, name='conv7')
            representation = conv_conv_shrink(outputs, [self.z_dim], self.trainable, name='conv8')
            print("after encoder:", representation.shape)
            outputs = tf.layers.dense(tf.reshape(representation, [-1, self.z_dim]), 1, name='dense_custom')
            return tf.nn.sigmoid(outputs), representation

    def build_decoder(self, z):
        print("-----------build decoder-----------")
        with tf.variable_scope('decoder') as scope:
            outputs = tf.layers.dense(z, 4*4*1024)
            outputs = tf.reshape(outputs, [-1, 4, 4, 1024])
            outputs = tf.layers.conv2d_transpose(outputs, 512, 2, 2, name='decode_2')
            outputs = tf.layers.conv2d_transpose(outputs, 256, 2, 2, name='decode_3')
            outputs = tf.layers.conv2d_transpose(outputs, 128, 2, 2, name='decode_4')
            outputs = tf.layers.conv2d_transpose(outputs, 64, 2, 2, name='decode_5')
            outputs = tf.layers.conv2d_transpose(outputs, 32, 2, 2, name='decode_6')
            outputs = tf.layers.conv2d_transpose(outputs, 3, 2, 2, name='decode_7')
            print("after decoder:", outputs.shape)
        return outputs

    def make_update_ops(self, x_name, z_name):
        tf.add_to_collection('inputs', self.inputs[x_name])
        tf.add_to_collection('inputs', self.inputs[z_name])
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    def make_optimizer(self):
        optm_g = tf.train.AdamOptimizer(self.learning_rate).minimize(self.g_loss,
                                                   var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                                              scope='decoder'),
                                                   global_step=self.global_step)
        optm_d = tf.train.AdamOptimizer(self.learning_rate).minimize(self.d_loss,
                                                   var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                                              scope='encoder'),
                                                   global_step=self.global_step)
        self.optimizer = {'d': optm_d, 'g': optm_g}

    def make_summary(self):
        tf.summary.scalar('d loss', self.d_loss)
        tf.summary.scalar('g loss', self.g_loss)
        tf.summary.scalar('learning rate', self.learning_rate)
        self.summary = tf.summary.merge_all()

    def train_config(self, x_name, y_name, n_train, n_valid, patch_size, ckdir):
        self.make_loss()
        self.make_learning_rate(n_train)
        self.make_update_ops(x_name, y_name)
        self.make_optimizer()
        self.make_ckdir(ckdir, patch_size)
        self.make_summary()
        self.config = tf.ConfigProto()
        self.n_train = n_train
        self.n_valid = n_valid

    def make_loss(self):
        self.g_loss = -tf.reduce_mean(tf.log(self.discr_f))
        self.d_loss = -tf.reduce_mean(tf.log(self.discr_r) + tf.log(1. - self.discr_f))

    def train(self, x_name, y_name, n_train, sess, summary_writer, n_valid=1000,
              train_reader=None, valid_reader=None,
              verb_step=100, save_epoch=5,
              img_mean=np.array((0, 0, 0), dtype=np.float32),
              continue_dir=None, valid_iou=False):
        # define summary operations
        valid_g_summary_op = tf.summary.scalar('g_loss_validation', self.valid_cross_entropy)
        valid_d_summary_op = tf.summary.scalar('d_loss_validation', self.valid_cross_entropy)
        valid_image_summary_op = tf.summary.image('Validation_images_summary', self.valid_images,
                                                  max_outputs=10)
        img_dir, _ = utils.get_task_img_folder()
        img_dir = os.path.join(img_dir, self.model_name, 'valid_imgs')

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
            for step in range(start_step, n_train, self.bs):
                X_batch, _ = train_reader.readerAction(sess)
                Z_batch = np.random.uniform(-1, 1, [self.bs, self.z_dim])
                _, self.global_step_value = sess.run([self.optimizer['g'], self.global_step],
                                                     feed_dict={self.inputs[x_name]:X_batch,
                                                                self.inputs[y_name]:Z_batch,
                                                                self.trainable: True})
                X_batch, _ = train_reader.readerAction(sess)
                Z_batch = np.random.uniform(-1, 1, [self.bs, self.z_dim])
                _, self.global_step_value = sess.run([self.optimizer['d'], self.global_step],
                                                     feed_dict={self.inputs[x_name]: X_batch,
                                                                self.inputs[y_name]: Z_batch,
                                                                self.trainable: True})

                if self.global_step_value % verb_step == 0:
                    d_loss, g_loss, step_summary = sess.run([self.d_loss, self.g_loss, self.summary],
                                                    feed_dict={self.inputs[x_name]: X_batch,
                                                               self.inputs[y_name]: Z_batch,
                                                               self.trainable: False})
                    summary_writer.add_summary(step_summary, self.global_step_value)
                    print('Epoch {:d} step {:d}\td_loss = {:.3f}, g_loss = {:.3f}'.
                          format(epoch, self.global_step_value, d_loss, g_loss))
            # validation
            loss_valid_mean = []
            g_loss_val_mean = []
            d_loss_val_mean = []
            for step in range(0, n_valid, self.bs):
                X_batch_val, _ = valid_reader.readerAction(sess)
                Z_batch_val = np.random.uniform(-1, 1, [self.bs, self.z_dim])
                d_loss_val, g_loss_val = sess.run([self.d_loss, self.g_loss],
                                                  feed_dict={self.inputs[x_name]: X_batch_val,
                                                             self.inputs[y_name]: Z_batch_val,
                                                             self.trainable: False})
                loss_valid_mean.append(d_loss_val+g_loss_val)
                g_loss_val_mean.append(g_loss_val)
                d_loss_val_mean.append(d_loss_val)
            loss_valid_mean = np.mean(loss_valid_mean)
            duration = time.time() - start_time
            print('Validation loss: {:.3f}, duration: {:.3f}'.format(loss_valid_mean, duration))
            valid_g_summary = sess.run(valid_g_summary_op,
                                       feed_dict={self.valid_cross_entropy: np.mean(g_loss_val_mean)})
            valid_d_summary = sess.run(valid_d_summary_op,
                                       feed_dict={self.valid_cross_entropy: np.mean(d_loss_val_mean)})
            summary_writer.add_summary(valid_g_summary, self.global_step_value)
            summary_writer.add_summary(valid_d_summary, self.global_step_value)
            if loss_valid_mean < loss_valid_min:
                loss_valid_min = loss_valid_mean
                saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)
                saver.save(sess, '{}/best_model.ckpt'.format(self.ckdir))

            valid_img_gen = sess.run(self.gener, feed_dict={self.inputs[y_name]: Z_batch_val})
            valid_image_summary = sess.run(valid_image_summary_op,
                                           feed_dict={self.valid_images: valid_img_gen[:,:,:,:3]+img_mean})
            summary_writer.add_summary(valid_image_summary, self.global_step_value)
            img2save = make_thumbnail(valid_img_gen, 1, 2, 5)
            if not os.path.exists(img_dir):
                os.makedirs(img_dir)
            img_name = os.path.join(img_dir, '{}_{}.png'.format(self.model_name, self.global_step_value))
            imageio.imsave(img_name, img2save)

            if epoch % save_epoch == 0:
                saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)
                saver.save(sess, '{}/model_{}.ckpt'.format(self.ckdir, epoch), global_step=self.global_step)

    def run(self, train_reader=None, valid_reader=None, test_reader=None, pretrained_model_dir=None, layers2load=None,
            isTrain=False, img_mean=np.array((0, 0, 0), dtype=np.float32), verb_step=100, save_epoch=5, gpu=None,
            tile_size=(5000, 5000), patch_size=(572, 572), truth_val=1, continue_dir=None, load_epoch_num=None,
            fineTune=False, valid_iou=False):
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
                        if not fineTune:
                            restore_var = [v for v in tf.global_variables() if 'resnet_v1' in v.name and
                                       not 'Adam' in v.name and not 'custom' in v.name]
                            loader = tf.train.Saver(var_list=restore_var)
                            self.load(pretrained_model_dir, sess, loader, epoch=load_epoch_num)
                        else:
                            self.load(pretrained_model_dir, sess, saver, epoch=load_epoch_num)
                threads = tf.train.start_queue_runners(coord=coord, sess=sess)
                try:
                    train_summary_writer = tf.summary.FileWriter(self.ckdir, sess.graph)
                    self.train('X', 'Z', self.n_train, sess, train_summary_writer,
                               n_valid=self.n_valid, train_reader=train_reader, valid_reader=valid_reader,
                               img_mean=img_mean,
                               verb_step=verb_step, save_epoch=save_epoch, continue_dir=continue_dir,
                               valid_iou=valid_iou)
                finally:
                    coord.request_stop()
                    coord.join(threads)
                    saver.save(sess, '{}/model.ckpt'.format(self.ckdir), global_step=self.global_step)
        else:
            '''with tf.Session() as sess:
                init = tf.global_variables_initializer()
                sess.run(init)
                self.load(pretrained_model_dir, sess, epoch=load_epoch_num)
                self.model_name = pretrained_model_dir.split('/')[-1]
                result = self.test('X', sess, test_reader)
            image_pred = uabUtilreader.un_patchify(result, tile_size, patch_size)
            return util_functions.get_pred_labels(image_pred) * truth_val'''
            pass


def image_summary(prediction, img_mean=np.array((0, 0, 0), dtype=np.float32)):
    return (prediction+img_mean).astype(np.uint8)


class VGGGAN(uabMakeNetwork_UNetEncoder.VGGVAE):
    def __init__(self, inputs, trainable, input_size, model_name='', dropout_rate=None,
                 learn_rate=1e-4, decay_step=60, decay_rate=0.1, epochs=100,
                 batch_size=5, start_filter_num=32, latent_num=500):
        network.Network.__init__(self, inputs, trainable, dropout_rate,
                                 learn_rate, decay_step, decay_rate, epochs, batch_size)
        self.name = 'VGGGAN'
        self.model_name = self.get_unique_name(model_name)
        self.sfn = start_filter_num
        self.learning_rate = None
        self.valid_d_summary = tf.placeholder(tf.float32, [])
        self.valid_g_summary = tf.placeholder(tf.float32, [])
        self.update_ops = None
        self.config = None
        self.n_train = 0
        self.n_valid = 0
        self.latent_num = latent_num
        self.valid_images = tf.placeholder(tf.uint8, shape=[None, input_size[0],
                                                            input_size[1], 3], name='validation_images')

    def create_graph(self, x_name, class_num, start_filter_num=32):
        self.class_num = class_num
        self.input_size = self.inputs[x_name].shape[1:3]

        self.gener = self.build_generator(tf.reshape(self.inputs['Z'], [-1, self.latent_num]))

        self.discr_f = self.build_discriminator(self.gener, None)
        self.discr_r = self.build_discriminator(self.inputs[x_name], True)

    def build_discriminator(self, x, reuse):
        with tf.variable_scope('encoder', reuse=reuse):
            conv1 = self.conv_conv_pool(x, [self.sfn], self.trainable, name='conv1',
                                        conv_stride=(2, 2), padding='same', dropout=self.dropout_rate,
                                        pool=False, activation=tf.nn.relu)  # 16*128*128
            conv2 = self.conv_conv_pool(conv1, [2 * self.sfn], self.trainable, name='conv2',
                                        conv_stride=(2, 2), padding='same', dropout=self.dropout_rate,
                                        pool=False, activation=tf.nn.relu)  # 32*64*64
            conv3 = self.conv_conv_pool(conv2, [4 * self.sfn], self.trainable, name='conv3',
                                        conv_stride=(2, 2), padding='same', dropout=self.dropout_rate,
                                        pool=False, activation=tf.nn.relu)  # 64*32*32
            conv4 = self.conv_conv_pool(conv3, [8 * self.sfn], self.trainable, name='conv4',
                                        conv_stride=(2, 2), padding='same', dropout=self.dropout_rate,
                                        pool=False, activation=tf.nn.relu)  # 128*16*16
            conv5 = self.conv_conv_pool(conv4, [16 * self.sfn], self.trainable, name='conv5',
                                        conv_stride=(2, 2), padding='same', dropout=self.dropout_rate,
                                        pool=False, activation=tf.nn.relu)  # 256*8*8
            conv6 = self.conv_conv_pool(conv5, [32 * self.sfn], self.trainable, name='conv6',
                                        conv_stride=(2, 2), padding='same', dropout=self.dropout_rate,
                                        pool=False, activation=tf.nn.relu)  # 512*4*4
            conv6_flat = tf.reshape(conv6, [-1, 32 * self.sfn * 4 * 4])
            self.representation = self.fc_fc(conv6_flat, [self.latent_num], self.trainable, 'encoding',
                                        activation=tf.nn.relu, dropout=False)
            return self.fc_fc(self.representation, [1], self.trainable, 'discriminate', activation=tf.nn.sigmoid,
                              dropout=False)

    def build_generator(self, z):
        with tf.variable_scope('decoder'):
            up0 = self.fc_fc(z, [512 * 4 * 4], self.trainable, 'decode_z', activation=None, dropout=False)
            up0 = tf.reshape(up0, [-1, 4, 4, 512])

            up1 = self.upsampling_2D(up0, 'upsample_0')  # 512*8*8
            conv6 = self.conv_conv_pool(up1, [self.sfn * 16], self.trainable, name='conv6',
                                        padding='same', dropout=self.dropout_rate, pool=False,
                                        activation=tf.nn.relu)  # 256*8*8
            up2 = self.upsampling_2D(conv6, 'upsample_1')  # 256*16*16
            conv7 = self.conv_conv_pool(up2, [self.sfn * 8], self.trainable, name='conv7',
                                        padding='same', dropout=self.dropout_rate, pool=False,
                                        activation=tf.nn.relu)  # 128*16*16
            up3 = self.upsampling_2D(conv7, 'upsample_2')  # 128*32*32
            conv8 = self.conv_conv_pool(up3, [self.sfn * 4], self.trainable, name='conv8',
                                        padding='same', dropout=self.dropout_rate, pool=False,
                                        activation=tf.nn.relu)  # 64*32*32
            up4 = self.upsampling_2D(conv8, 'upsample_3')  # 64*64*64
            conv9 = self.conv_conv_pool(up4, [self.sfn * 2], self.trainable, name='conv9',
                                        padding='same', dropout=self.dropout_rate, pool=False,
                                        activation=tf.nn.relu)  # 32*64*64
            up5 = self.upsampling_2D(conv9, 'upsample_4')  # 32*128*128
            conv10 = self.conv_conv_pool(up5, [self.sfn], self.trainable, name='conv10',
                                         padding='same', dropout=self.dropout_rate, pool=False,
                                         activation=tf.nn.relu)  # 16*128*128
            up6 = self.upsampling_2D(conv10, 'upsample_5')  # 16*256*256
            outputs = tf.layers.conv2d(up6, self.class_num, (3, 3), name='final', activation=None, padding='same')
        return outputs

    def make_loss(self, y_name, loss_type='xent', **kwargs):
        self.g_loss = -tf.reduce_mean(tf.log(self.discr_f))
        self.d_loss = -tf.reduce_mean(tf.log(self.discr_r) + tf.log(1. - self.discr_f))

    def make_optimizer(self, train_var_filter):
        optm_g = tf.train.AdamOptimizer(self.learning_rate).minimize(self.g_loss,
                                                                     var_list=tf.get_collection(
                                                                         tf.GraphKeys.GLOBAL_VARIABLES,
                                                                         scope='decoder'),
                                                                     global_step=self.global_step)
        optm_d = tf.train.AdamOptimizer(self.learning_rate).minimize(self.d_loss,
                                                                     var_list=tf.get_collection(
                                                                         tf.GraphKeys.GLOBAL_VARIABLES,
                                                                         scope='encoder'),
                                                                     global_step=self.global_step)
        self.optimizer = {'d': optm_d, 'g': optm_g}

    def make_summary(self, hist=False):
        tf.summary.scalar('d loss', self.d_loss)
        tf.summary.scalar('g loss', self.g_loss)
        tf.summary.scalar('learning rate', self.learning_rate)
        self.summary = tf.summary.merge_all()

    def train_config(self, x_name, y_name, n_train, n_valid, patch_size, ckdir, loss_type='xent', train_var_filter=None,
                     hist=False, **kwargs):
        self.make_loss(y_name, loss_type, **kwargs)
        self.make_learning_rate(n_train)
        self.make_update_ops(x_name, y_name)
        self.make_optimizer(train_var_filter)
        self.make_ckdir(ckdir, patch_size)
        self.make_summary()
        self.config = tf.ConfigProto()
        self.n_train = n_train
        self.n_valid = n_valid

    def make_update_ops(self, x_name, z_name):
        tf.add_to_collection('inputs', self.inputs[x_name])
        tf.add_to_collection('inputs', self.inputs[z_name])
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    def train(self, x_name, y_name, n_train, sess, summary_writer, n_valid=1000,
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
            for step in range(start_step, n_train, self.bs):
                X_batch, _ = train_reader.readerAction(sess)
                Z_batch = np.random.uniform(-1, 1, [self.bs, self.latent_num])
                _, self.global_step_value = sess.run([self.optimizer['g'], self.global_step],
                                                     feed_dict={self.inputs[x_name]:X_batch,
                                                                self.inputs[y_name]:Z_batch,
                                                                self.trainable: True})
                X_batch, _ = train_reader.readerAction(sess)
                Z_batch = np.random.uniform(-1, 1, [self.bs, self.latent_num])
                _, self.global_step_value = sess.run([self.optimizer['d'], self.global_step],
                                                     feed_dict={self.inputs[x_name]: X_batch,
                                                               self.inputs[y_name]: Z_batch,
                                                               self.trainable: True})

                if self.global_step_value % verb_step == 0:
                    d_loss, g_loss, step_summary = sess.run([self.d_loss, self.g_loss, self.summary],
                                                    feed_dict={self.inputs[x_name]: X_batch,
                                                               self.inputs[y_name]: Z_batch,
                                                               self.trainable: False})
                    summary_writer.add_summary(step_summary, self.global_step_value)
                    print('Epoch {:d} step {:d}\td_loss = {:.3f}, g_loss = {:.3f}'.
                          format(epoch, self.global_step_value, d_loss, g_loss))
            # validation
            loss_valid_mean = []
            g_loss_val_mean = []
            d_loss_val_mean = []
            for step in range(0, n_valid, self.bs):
                X_batch_val, _ = valid_reader.readerAction(sess)
                Z_batch_val = np.random.uniform(-1, 1, [self.bs, self.latent_num])
                d_loss_val, g_loss_val = sess.run([self.d_loss, self.g_loss],
                                                  feed_dict={self.inputs[x_name]: X_batch_val,
                                                             self.inputs[y_name]: Z_batch_val,
                                                             self.trainable: False})
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

            valid_img_gen = sess.run(self.gener, feed_dict={self.inputs[y_name]: Z_batch_val,
                                                            self.trainable: False})
            if image_summary is not None:
                valid_image_summary = sess.run(valid_image_summary_op,
                                               feed_dict={self.valid_images:
                                                              image_summary(valid_img_gen, img_mean)})
                summary_writer.add_summary(valid_image_summary, self.global_step_value)

            if epoch % save_epoch == 0:
                saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)
                saver.save(sess, '{}/model_{}.ckpt'.format(self.ckdir, epoch), global_step=self.global_step)

    def run(self, train_reader=None, valid_reader=None, test_reader=None, pretrained_model_dir=None, layers2load=None,
            isTrain=False, img_mean=np.array((0, 0, 0), dtype=np.float32), verb_step=100, save_epoch=5, gpu=None,
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
                    self.train('X', 'Z', self.n_train, sess, train_summary_writer,
                               n_valid=self.n_valid, train_reader=train_reader, valid_reader=valid_reader,
                               image_summary=image_summary, img_mean=img_mean,
                               verb_step=verb_step, save_epoch=save_epoch, continue_dir=continue_dir,
                               valid_iou=valid_iou)
                finally:
                    coord.request_stop()
                    coord.join(threads)
                    saver.save(sess, '{}/model.ckpt'.format(self.ckdir), global_step=self.global_step)


def make_thumbnail(imgs, mult, row, col):
    n, w, h, _ = imgs.shape
    w_s = int(w / mult)
    h_s = int(h / mult)
    thumbnail = np.zeros((w_s*row, h_s*col, 3), dtype=np.uint8)
    for i in range(10):
        n_row = int(i // col)
        n_col = int(i % col)

        thumbnail[n_row*w_s:(n_row+1)*w_s, n_col*h_s:(n_col+1)*h_s, :] = \
            scipy.misc.imresize(imgs[i, :, :, :], 1/mult).astype(np.uint8)
    return thumbnail
