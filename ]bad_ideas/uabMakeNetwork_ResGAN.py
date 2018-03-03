import os
import time
import imageio
import numpy as np
import scipy.misc
import tensorflow as tf
import utils
from bohaoCustom import uabMakeNetwork as network
from bohaoCustom import uabMakeNetwork_DeepLabV2

class ResGAN(uabMakeNetwork_DeepLabV2.DeeplabV3):
    def __init__(self, inputs, trainable, input_size, model_name='', dropout_rate=0.2,
                 learn_rate=1e-4, decay_step=60, decay_rate=0.1, epochs=100,
                 batch_size=5, start_filter_num=32, z_dim=100):
        network.Network.__init__(self, inputs, trainable, dropout_rate,
                                 learn_rate, decay_step, decay_rate, epochs, batch_size)
        self.name = 'ResGAN'
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
        scope_name = 'resnet_v1_101'
        with tf.variable_scope(scope_name, reuse=reuse) as scope:
            outputs = self._start_block('conv1', x)
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
                outputs = self._bottleneck_resblock(outputs, 1024, 'unit_1', half_size=True, identity_connection=False)
                num_layers_block3 = 23
                for i in range(2, num_layers_block3 + 1):
                    outputs = self._bottleneck_resblock(outputs, 1024, 'unit_%d' % i)
                print("after block3:", outputs.shape)
            with tf.variable_scope('block4') as scope:
                outputs = self._bottleneck_resblock(outputs, 2048, 'unit_1', half_size=True, identity_connection=False)
                outputs = self._bottleneck_resblock(outputs, 2048, 'unit_2')
                outputs = self._bottleneck_resblock(outputs, 2048, 'unit_3')
                print("after block4:", outputs.shape)
            with tf.variable_scope('block5_custom') as scope:
                outputs = self._bottleneck_resblock(outputs, 1024, 'unit_1', half_size=True, identity_connection=False)
                outputs = self._bottleneck_resblock(outputs, 512, 'unit_2', half_size=True, identity_connection=False)
                representation = self._bottleneck_resblock(outputs, self.z_dim, 'unit_3', half_size=True, identity_connection=False)
                print("after block5:", representation.shape)
            outputs = tf.layers.dense(tf.reshape(representation, [-1, self.z_dim]), 1, name='dense_custom')
            return tf.nn.sigmoid(outputs), representation

    def build_decoder(self, z):
        print("-----------build decoder-----------")
        with tf.variable_scope('decoder') as scope:
            #outputs = tf.layers.conv2d_transpose(z, 1024, 4, 2, name='decode_1')
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

    def _start_block(self, name, x):
        outputs = self._conv2d(x, 7, 64, 2, name=name)
        outputs = self._batch_norm(outputs, name=name, is_training=False, activation_fn=tf.nn.relu)
        outputs = self._max_pool2d(outputs, 3, 2, name='pool1')
        return outputs

    def make_update_ops(self, x_name, z_name):
        tf.add_to_collection('inputs', self.inputs[x_name])
        tf.add_to_collection('inputs', self.inputs[z_name])
        # tf.add_to_collection('outputs', self.pred)
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    def make_optimizer(self):
        optm_g = tf.train.AdamOptimizer(self.learning_rate).minimize(self.g_loss,
                                                   var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                                              scope='decoder'),
                                                   global_step=self.global_step)
        optm_d = tf.train.AdamOptimizer(self.learning_rate).minimize(self.d_loss,
                                                   var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                                              scope='resnet_v1_101'),
                                                   global_step=self.global_step)
        self.optimizer = {'d': optm_d, 'g': optm_g}

    def make_summary(self):
        tf.summary.scalar('d loss', self.d_loss)
        tf.summary.scalar('g loss', self.d_loss)
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


def make_thumbnail(imgs, mult, row, col):
    n, w, h, _ = imgs.shape
    w_s = int(w / mult)
    h_s = int(h / mult)
    thumbnail = np.zeros((w_s*row, h_s*col, 3), dtype=np.uint8)
    for i in range(n):
        n_row = int(i // col)
        n_col = int(i % col)

        thumbnail[n_row*w_s:(n_row+1)*w_s, n_col*h_s:(n_col+1)*h_s, :] = \
            scipy.misc.imresize(imgs[i, :, :, :], 1/mult).astype(np.uint8)
    return thumbnail
