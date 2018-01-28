import os
import time
import imageio
import numpy as np
import tensorflow as tf
import util_functions
import uabUtilreader
import uabDataReader
import uabRepoPaths
from bohaoCustom import uabMakeNetwork as network


class UnetModel(network.Network):
    def __init__(self, inputs, trainable, input_size, model_name='', dropout_rate=0.2,
                 learn_rate=1e-4, decay_step=60, decay_rate=0.1, epochs=100,
                 batch_size=5, start_filter_num=32):
        network.Network.__init__(self, inputs, trainable, dropout_rate,
                                 learn_rate, decay_step, decay_rate, epochs, batch_size)
        self.name = 'Unet'
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

    def make_ckdir(self, ckdir, patch_size):
        if type(patch_size) is list:
            patch_size = patch_size[0]
        # make unique directory for save
        dir_name = '{}_PS{}_BS{}_EP{}_LR{}_DS{}_DR{}_SFN{}'.\
            format(self.model_name, patch_size, self.bs, self.epochs, self.lr, self.ds, self.dr, self.sfn)
        self.ckdir = os.path.join(ckdir, dir_name)

    def create_graph(self, x_name, class_num):
        self.class_num = class_num
        sfn = self.sfn

        # downsample
        conv1, pool1 = self.conv_conv_pool(self.inputs[x_name], [sfn, sfn], self.trainable, name='conv1')
        conv2, pool2 = self.conv_conv_pool(pool1, [sfn*2, sfn*2], self.trainable, name='conv2')
        conv3, pool3 = self.conv_conv_pool(pool2, [sfn*4, sfn*4], self.trainable, name='conv3')
        conv4, pool4 = self.conv_conv_pool(pool3, [sfn*8, sfn*8], self.trainable, name='conv4')
        conv5 = self.conv_conv_pool(pool4, [sfn*16, sfn*16], self.trainable, name='conv5', pool=False)

        # upsample
        up6 = self.upsample_concat(conv5, conv4, name='6')
        conv6 = self.conv_conv_pool(up6, [sfn*8, sfn*8], self.trainable, name='up6', pool=False)
        up7 = self.upsample_concat(conv6, conv3, name='7')
        conv7 = self.conv_conv_pool(up7, [sfn*4, sfn*4], self.trainable, name='up7', pool=False)
        up8 = self.upsample_concat(conv7, conv2, name='8')
        conv8 = self.conv_conv_pool(up8, [sfn*2, sfn*2], self.trainable, name='up8', pool=False)
        up9 = self.upsample_concat(conv8, conv1, name='9')
        conv9 = self.conv_conv_pool(up9, [sfn, sfn], self.trainable, name='up9', pool=False)

        self.pred = tf.layers.conv2d(conv9, class_num, (1, 1), name='final', activation=None, padding='same')
        self.output = tf.nn.softmax(self.pred)

    def load_weights(self, ckpt_dir, layers2load):
        # this is different from network.load()
        # this function only loads specified layers
        layers_list = []
        if isinstance(layers2load, str):
            layers2load = [int(a) for a in layers2load.split(',')]
        for layer_id in layers2load:
            assert 1 <= layer_id <= 9
            if layer_id <= 5:
                prefix = 'layerconv'
            else:
                prefix = 'layerup'
            layers_list.append('{}{}'.format(prefix, layer_id))

        load_dict = {}
        for layer_name in layers_list:
            feed_layer = layer_name + '/'
            load_dict[feed_layer] = feed_layer
        tf.contrib.framework.init_from_checkpoint(ckpt_dir, load_dict)

    def restore_model(self,sess):
        # automatically restore last saved model if checkpoint exists
        if tf.train.latest_checkpoint(self.ckdir): 

            self.load(self.ckdir,sess)

            with open(os.path.join(self.ckdir,'checkpoint'),'r') as f:
                model_checkpoint_path = f.readline().split('/')[-1]
            buf = [int(i) for i in re.findall(r"\d+", model_checkpoint_path)]
            if len(buf) == 1:
                start_step = buf[0]+1
                self.start_epoch = int(np.floor(start_step/(8000/self.bs)))
            elif len(buf) == 2:
                self.start_epoch = buf[0]+1
                start_step = buf[1]+1
        else:
            self.start_epoch,start_step = [0,0]
            
        sess.run(self.global_step.assign(start_step))
        self.global_step_value = self.global_step.eval()
        print('restoring model from epoch %d step %d'%(self.start_epoch,self.global_step_value))

    def make_learning_rate(self, n_train):
        self.learning_rate = tf.train.exponential_decay(self.lr, self.global_step,
                                                        tf.cast(n_train/self.bs * self.ds, tf.int32),
                                                        self.dr, staircase=True)

    def make_loss(self, y_name, loss_type='xent'):
        # TODO loss type IoU
        with tf.variable_scope('loss'):
            pred_flat = tf.reshape(self.pred, [-1, self.class_num])
            y_flat = tf.reshape(tf.squeeze(self.inputs[y_name], axis=[3]), [-1, ])
            indices = tf.squeeze(tf.where(tf.less_equal(y_flat, self.class_num - 1)), 1)
            gt = tf.gather(y_flat, indices)
            prediction = tf.gather(pred_flat, indices)
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt))

    def make_update_ops(self, x_name, y_name):
        tf.add_to_collection('inputs', self.inputs[x_name])
        tf.add_to_collection('inputs', self.inputs[y_name])
        tf.add_to_collection('outputs', self.pred)
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    def make_optimizer(self):
        with tf.control_dependencies(self.update_ops):
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)

    def make_summary(self):
        tf.summary.histogram('Predicted Prob', tf.argmax(tf.nn.softmax(self.pred), 1))
        tf.summary.scalar('Cross Entropy', self.loss)
        tf.summary.scalar('learning rate', self.learning_rate)
        self.summary = tf.summary.merge_all()

    def train_config(self, x_name, y_name, n_train, n_valid, patch_size, ckdir, loss_type='xent'):
        self.make_loss(y_name, loss_type)
        self.make_learning_rate(n_train)
        self.make_update_ops(x_name, y_name)
        self.make_optimizer()
        self.make_ckdir(ckdir, patch_size)
        self.make_summary()
        self.config = tf.ConfigProto()
        self.n_train = n_train
        self.n_valid = n_valid

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

        for epoch in range(start_epoch, self.epochs):
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
            print('Validation cross entropy: {:.3f}'.format(cross_entropy_valid_mean))
            valid_cross_entropy_summary = sess.run(valid_cross_entropy_summary_op,
                                                   feed_dict={self.valid_cross_entropy: cross_entropy_valid_mean})
            summary_writer.add_summary(valid_cross_entropy_summary, self.global_step_value)

            if image_summary is not None:
                valid_image_summary = sess.run(valid_image_summary_op,
                                               feed_dict={self.valid_images:
                                                              image_summary(X_batch_val[:,:,:,:3], y_batch_val, pred_valid,
                                                                            img_mean)})
                summary_writer.add_summary(valid_image_summary, self.global_step_value)

            if epoch % save_epoch == 0:
                saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)
                saver.save(sess, '{}/model_{}.ckpt'.format(self.ckdir, epoch), global_step=self.global_step)

    def test(self, x_name, sess, test_iterator):
        result = []
        for X_batch in test_iterator:
            #pred = sess.run(tf.nn.softmax(self.pred), feed_dict={self.inputs[x_name]:X_batch,
            #                                                     self.trainable: False})
            pred = sess.run(self.output, feed_dict={self.inputs[x_name]: X_batch,
                                                    self.trainable: False})
            result.append(pred)
        result = np.vstack(result)
        return result

    def get_overlap(self):
        # TODO calculate the padding pixels
        return 0

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
                        self.load(pretrained_model_dir, sess, saver)
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

    def evaluate(self, rgb_list, gt_list, rgb_dir, gt_dir, input_size, tile_size, batch_size, img_mean,
                 model_dir, gpu=None, save_result=True, save_result_parent_dir=None, show_figure=False,
                 verb=True, ds_name='default'):
        if show_figure:
            import matplotlib.pyplot as plt
        iou_record = []
        iou_return = {}
        for file_name, file_name_truth in zip(rgb_list, gt_list):
            tile_name = file_name_truth.split('_')[0]
            if verb:
                print('Evaluating {} ... '.format(tile_name))
            start_time = time.time()

            # prepare the reader
            reader = uabDataReader.ImageLabelReader(gtInds=[0],
                                                    dataInds=[0],
                                                    nChannels=3,
                                                    parentDir=rgb_dir,
                                                    chipFiles=[file_name],
                                                    chip_size=input_size,
                                                    tile_size=tile_size,
                                                    batchSize=batch_size,
                                                    block_mean=img_mean,
                                                    overlap=self.get_overlap(),
                                                    padding=np.array((self.get_overlap()/2, self.get_overlap()/2)),
                                                    isTrain=False)
            rManager = reader.readManager

            # run the model
            pred = self.run(pretrained_model_dir=model_dir,
                            test_reader=rManager,
                            tile_size=tile_size,
                            patch_size=input_size,
                            gpu=gpu)

            truth_label_img = imageio.imread(os.path.join(gt_dir, file_name_truth))
            iou = util_functions.iou_metric(truth_label_img, pred, divide_flag=True)
            iou_record.append(iou)
            iou_return[tile_name] = iou

            duration = time.time() - start_time
            if verb:
                print('{} mean IoU={:.3f}, duration: {:.3f}'.format(tile_name, iou[0]/iou[1], duration))

            # save results
            if save_result:
                if save_result_parent_dir is None:
                    score_save_dir = os.path.join(uabRepoPaths.evalPath, self.model_name, ds_name)
                else:
                    score_save_dir = os.path.join(uabRepoPaths.evalPath, save_result_parent_dir,
                                                  self.model_name, ds_name)
                pred_save_dir = os.path.join(score_save_dir, 'pred')
                if not os.path.exists(score_save_dir):
                    os.makedirs(score_save_dir)
                if not os.path.exists(pred_save_dir):
                    os.makedirs(pred_save_dir)
                imageio.imwrite(os.path.join(pred_save_dir, tile_name+'.png'), pred)
                with open(os.path.join(score_save_dir, 'result.txt'), 'a+') as file:
                    file.write('{} {}\n'.format(tile_name, iou))

            if show_figure:
                plt.figure(figsize=(12, 4))
                ax1 = plt.subplot(121)
                ax1.imshow(truth_label_img)
                plt.title('Truth')
                ax2 = plt.subplot(122, sharex=ax1, sharey=ax1)
                ax2.imshow(pred)
                plt.title('pred')
                plt.suptitle('{} Results on {} IoU={:3f}'.format(self.model_name, file_name_truth.split('_')[0], iou))
                plt.show()

        mean_iou = np.mean([x[0]/x[1] for x in iou_record])
        print('Overall mean IoU={:.3f}'.format(mean_iou))
        if save_result:
            if save_result_parent_dir is None:
                score_save_dir = os.path.join(uabRepoPaths.evalPath, self.model_name, ds_name)
            else:
                score_save_dir = os.path.join(uabRepoPaths.evalPath, save_result_parent_dir, self.model_name,
                                              ds_name)
            with open(os.path.join(score_save_dir, 'result.txt'), 'a+') as file:
                file.write('{}'.format(mean_iou))

        return iou_return



class UnetModelCrop(UnetModel):
    def __init__(self, inputs, trainable, input_size, model_name='', dropout_rate=0.2,
                 learn_rate=1e-4, decay_step=60, decay_rate=0.1, epochs=100,
                 batch_size=5, start_filter_num=32):
        network.Network.__init__(self, inputs, trainable, dropout_rate,
                                 learn_rate, decay_step, decay_rate, epochs, batch_size)
        self.name = 'UnetCrop'
        self.model_name = self.get_unique_name(model_name)
        self.sfn = start_filter_num
        self.learning_rate = None
        self.valid_cross_entropy = tf.placeholder(tf.float32, [])
        self.valid_images = tf.placeholder(tf.uint8, shape=[None, input_size[0],
                                                            input_size[1] * 3, 3], name='validation_images')
        self.update_ops = None

    def create_graph(self, x_name, class_num, start_filter_num=32):
        self.class_num = class_num
        sfn = self.sfn

        # downsample
        conv1, pool1 = self.conv_conv_pool(self.inputs[x_name], [sfn, sfn], self.trainable, name='conv1', padding='valid')
        conv2, pool2 = self.conv_conv_pool(pool1, [sfn*2, sfn*2], self.trainable, name='conv2', padding='valid')
        conv3, pool3 = self.conv_conv_pool(pool2, [sfn*4, sfn*4], self.trainable, name='conv3', padding='valid')
        conv4, pool4 = self.conv_conv_pool(pool3, [sfn*8, sfn*8], self.trainable, name='conv4', padding='valid')
        conv5 = self.conv_conv_pool(pool4, [sfn*16, sfn*16], self.trainable, name='conv5', pool=False, padding='valid')

        # upsample
        up6 = self.crop_upsample_concat(conv5, conv4, 8, name='6')
        conv6 = self.conv_conv_pool(up6, [sfn*8, sfn*8], self.trainable, name='up6', pool=False, padding='valid')
        up7 = self.crop_upsample_concat(conv6, conv3, 32, name='7')
        conv7 = self.conv_conv_pool(up7, [sfn*4, sfn*4], self.trainable, name='up7', pool=False, padding='valid')
        up8 = self.crop_upsample_concat(conv7, conv2, 80, name='8')
        conv8 = self.conv_conv_pool(up8, [sfn*2, sfn*2], self.trainable, name='up8', pool=False, padding='valid')
        up9 = self.crop_upsample_concat(conv8, conv1, 176, name='9')
        conv9 = self.conv_conv_pool(up9, [sfn, sfn], self.trainable, name='up9', pool=False, padding='valid')

        self.pred = tf.layers.conv2d(conv9, class_num, (1, 1), name='final', activation=None, padding='same')
        self.output = tf.nn.softmax(self.pred)

    def make_loss(self, y_name, loss_type='xent'):
        with tf.variable_scope('loss'):
            pred_flat = tf.reshape(self.pred, [-1, self.class_num])
            _, w, h, _ = self.inputs[y_name].get_shape().as_list()
            y = tf.image.resize_image_with_crop_or_pad(self.inputs[y_name], w-self.get_overlap(), h-self.get_overlap())
            y_flat = tf.reshape(tf.squeeze(y, axis=[3]), [-1, ])
            indices = tf.squeeze(tf.where(tf.less_equal(y_flat, self.class_num - 1)), 1)
            gt = tf.gather(y_flat, indices)
            prediction = tf.gather(pred_flat, indices)
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt))

    def load_weights_append_first_layer(self, ckpt_dir, layers2load, conv1_weight, check_weight=False):
        # this functino load weights from pretrained model and add extra filters to first layer
        layers_list = []
        for layer_id in layers2load:
            assert 1 <= layer_id <= 9
            if layer_id == 1:
                layers_list.append('layerconv1/conv_1/bias:0')
                layers_list.append('layerconv1/bn_1')
                layers_list.append('layerconv1/conv_2')
                layers_list.append('layerconv1/bn_2')
                continue
            elif layer_id <= 5:
                prefix = 'layerconv'
            else:
                prefix = 'layerup'
            layers_list.append('{}{}'.format(prefix, layer_id))

        load_dict = {}
        for layer_name in layers_list:
            feed_layer = layer_name + '/'
            load_dict[feed_layer] = feed_layer
        tf.contrib.framework.init_from_checkpoint(ckpt_dir, load_dict)

        layerconv1_kernel = tf.trainable_variables()[0]
        assign_op = layerconv1_kernel.assign(conv1_weight)
        with tf.Session() as sess:
            sess.run(assign_op)
            weight = sess.run(layerconv1_kernel)

        if check_weight:
            import matplotlib.pyplot as plt
            _, _, c_num, _ = weight.shape
            for i in range(c_num):
                plt.subplot(321+i)
                plt.imshow(weight[:, :, i, :].reshape((16, 18)))
                plt.colorbar()
                plt.title(i)
            plt.show()

    def get_overlap(self):
        # TODO calculate the padding pixels
        return 184

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
                        self.load(pretrained_model_dir, sess, saver)
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
            pad = self.get_overlap()
            with tf.Session() as sess:
                init = tf.global_variables_initializer()
                sess.run(init)
                self.load(pretrained_model_dir, sess)
                self.model_name = pretrained_model_dir.split('/')[-1]
                result = self.test('X', sess, test_reader)
            image_pred = uabUtilreader.un_patchify_shrink(result,
                                                          [tile_size[0] + pad, tile_size[1] + pad],
                                                          tile_size,
                                                          patch_size,
                                                          [patch_size[0] - pad, patch_size[1] - pad],
                                                          overlap=pad)
            return util_functions.get_pred_labels(image_pred) * truth_val


class UnetModelMoreCrop(UnetModelCrop):
    def __init__(self, inputs, trainable, input_size, model_name='', dropout_rate=0.2,
                 learn_rate=1e-4, decay_step=60, decay_rate=0.1, epochs=100,
                 batch_size=5, start_filter_num=32):
        network.Network.__init__(self, inputs, trainable, dropout_rate,
                                 learn_rate, decay_step, decay_rate, epochs, batch_size)
        self.name = 'UnetMoreCrop'
        self.model_name = self.get_unique_name(model_name)
        self.sfn = start_filter_num
        self.learning_rate = None
        self.valid_cross_entropy = tf.placeholder(tf.float32, [])
        self.valid_images = tf.placeholder(tf.uint8, shape=[None, input_size[0],
                                                            input_size[1] * 3, 3], name='validation_images')
        self.update_ops = None

    def create_graph(self, x_name, class_num, start_filter_num=32):
        self.class_num = class_num
        sfn = self.sfn

        # downsample
        conv1, pool1 = self.conv_conv_pool(self.inputs[x_name], [sfn, sfn], self.trainable, name='conv1', padding='valid')
        conv2, pool2 = self.conv_conv_pool(pool1, [sfn*2, sfn*2], self.trainable, name='conv2', padding='valid')
        conv3, pool3 = self.conv_conv_pool(pool2, [sfn*4, sfn*4], self.trainable, name='conv3', padding='valid')
        conv4, pool4 = self.conv_conv_pool(pool3, [sfn*8, sfn*8], self.trainable, name='conv4', padding='valid')
        conv5 = self.conv_conv_pool(pool4, [sfn*16, sfn*16], self.trainable, name='conv5', pool=False, padding='valid')

        # upsample
        up6 = self.crop_upsample_concat(conv5, conv4, 8, name='6')
        conv6 = self.conv_conv_pool(up6, [sfn*8, sfn*8], self.trainable, name='up6', pool=False, padding='valid')
        up7 = self.crop_upsample_concat(conv6, conv3, 32, name='7')
        conv7 = self.conv_conv_pool(up7, [sfn*4, sfn*4], self.trainable, name='up7', pool=False, padding='valid')
        up8 = self.crop_upsample_concat(conv7, conv2, 80, name='8')
        conv8 = self.conv_conv_pool(up8, [sfn*2, sfn*2], self.trainable, name='up8', pool=False, padding='valid')
        up9 = self.crop_upsample_concat(conv8, conv1, 176, name='9')
        conv9 = self.conv_conv_pool(up9, [sfn, sfn], self.trainable, name='up9', pool=False, padding='valid')

        _, w, h, _ = conv9.get_shape().as_list()
        crop9 = tf.image.resize_image_with_crop_or_pad(conv9, w-40, h-40)

        self.pred = tf.layers.conv2d(crop9, class_num, (1, 1), name='final', activation=None, padding='same')
        self.output = tf.nn.softmax(self.pred)

    def get_overlap(self):
        # TODO calculate the padding pixels
        return 224



#%% UnetModelCropWeighted has loss function that assigns weights to imbalanced classes
class UnetModelCropWeighted(UnetModelCrop):
    
    def __init__(self, inputs, trainable, input_size, model_name='', dropout_rate=0.2,learn_rate=1e-4, decay_step=60, decay_rate=0.1, epochs=100,batch_size=5, start_filter_num=32, class_weights = [1,1]):
        super(UnetModelCropWeighted, self).__init__( inputs, trainable, input_size, model_name, dropout_rate,learn_rate, decay_step, decay_rate, epochs,batch_size, start_filter_num)
        self.cweights = tf.constant([w/class_weights[0] for w in class_weights], dtype = tf.float32)
        self.name = 'UnetCropWeighted'
        self.model_name = self.get_unique_name(model_name)

    def make_loss(self, y_name, loss_type='xent'):
        # TODO loss type IoU
        with tf.variable_scope('loss'):
            pred_flat = tf.reshape(self.pred, [-1, self.class_num])
            _, w, h, _ = self.inputs[y_name].get_shape().as_list()
            y = tf.image.resize_image_with_crop_or_pad(self.inputs[y_name], w-self.get_overlap(), h-self.get_overlap())
            y_flat = tf.reshape(tf.squeeze(y, axis=[3]), [-1, ])
            indices = tf.squeeze(tf.where(tf.less_equal(y_flat, self.class_num - 1)), 1)
            gt = tf.gather(y_flat, indices)
            prediction = tf.gather(pred_flat, indices)
            weights = tf.gather(self.cweights,gt)
            self.loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(logits=prediction, labels=gt, weights=weights))



class UnetModel_Appendix(UnetModelCrop):
    def __init__(self, inputs, trainable, input_size, model_name='', dropout_rate=0.2,
                 learn_rate=1e-4, decay_step=60, decay_rate=0.1, epochs=100,
                 batch_size=5, start_filter_num=32):
        network.Network.__init__(self, inputs, trainable, dropout_rate,
                                 learn_rate, decay_step, decay_rate, epochs, batch_size)
        self.name = 'UnetCropAppend'
        self.model_name = self.get_unique_name(model_name)
        self.sfn = start_filter_num
        self.learning_rate = None
        self.valid_cross_entropy = tf.placeholder(tf.float32, [])
        self.valid_images = tf.placeholder(tf.uint8, shape=[None, input_size[0],
                                                            input_size[1] * 3, 3], name='validation_images')
        self.update_ops = None

    def create_graph(self, x_name, class_num, start_filter_num=32):
        self.class_num = class_num
        sfn = start_filter_num

        # downsample
        conv1, pool1 = self.conv_conv_pool(self.inputs[x_name], [sfn, sfn], self.trainable, name='conv1', padding='valid')
        conv2, pool2 = self.conv_conv_pool(pool1, [sfn*2, sfn*2], self.trainable, name='conv2', padding='valid')
        conv3, pool3 = self.conv_conv_pool(pool2, [sfn*4, sfn*4], self.trainable, name='conv3', padding='valid')
        conv4, pool4 = self.conv_conv_pool(pool3, [sfn*8, sfn*8], self.trainable, name='conv4', padding='valid')
        conv5 = self.conv_conv_pool(pool4, [sfn*16, sfn*16], self.trainable, name='conv5', pool=False, padding='valid')

        # upsample
        up6 = self.crop_upsample_concat(conv5, conv4, 8, name='6')
        conv6 = self.conv_conv_pool(up6, [sfn*8, sfn*8], self.trainable, name='up6', pool=False, padding='valid')
        up7 = self.crop_upsample_concat(conv6, conv3, 32, name='7')
        conv7 = self.conv_conv_pool(up7, [sfn*4, sfn*4], self.trainable, name='up7', pool=False, padding='valid')
        up8 = self.crop_upsample_concat(conv7, conv2, 80, name='8')
        conv8 = self.conv_conv_pool(up8, [sfn*2, sfn*2], self.trainable, name='up8', pool=False, padding='valid')
        up9 = self.crop_upsample_concat(conv8, conv1, 176, name='9')
        conv9 = self.conv_conv_pool(up9, [sfn, sfn], self.trainable, name='up9', pool=False, padding='valid')

        conv10 = tf.layers.conv2d(conv9, sfn, (1, 1), name='second_final', padding='same')
        self.pred = tf.layers.conv2d(conv10, class_num, (1, 1), name='final', activation=None, padding='same')
        self.output = tf.nn.softmax(self.pred)


class ResUnetModel_Crop(UnetModelCrop):
    def create_graph(self, x_name, class_num, start_filter_num=32):
        self.class_num = class_num
        sfn = start_filter_num

        # downsample
        conv1, pool1 = self.conv_conv_identity_pool_crop(self.inputs[x_name], [sfn, sfn], self.trainable, name='conv1',
                                                         padding='valid')
        conv2, pool2 = self.conv_conv_identity_pool_crop(pool1, [sfn*2, sfn*2], self.trainable, name='conv2',
                                                         padding='valid')
        conv3, pool3 = self.conv_conv_identity_pool_crop(pool2, [sfn*4, sfn*4], self.trainable, name='conv3',
                                                         padding='valid')
        conv4, pool4 = self.conv_conv_identity_pool_crop(pool3, [sfn*8, sfn*8], self.trainable, name='conv4',
                                                         padding='valid')
        conv5 = self.conv_conv_identity_pool_crop(pool4, [sfn*16, sfn*16], self.trainable, name='conv5', pool=False,
                                                  padding='valid')

        # upsample
        up6 = self.crop_upsample_concat(conv5, conv4, 8, name='6')
        conv6 = self.conv_conv_identity_pool_crop(up6, [sfn*8, sfn*8], self.trainable, name='up6', pool=False,
                                                  padding='valid')
        up7 = self.crop_upsample_concat(conv6, conv3, 32, name='7')
        conv7 = self.conv_conv_identity_pool_crop(up7, [sfn*4, sfn*4], self.trainable, name='up7', pool=False,
                                                  padding='valid')
        up8 = self.crop_upsample_concat(conv7, conv2, 80, name='8')
        conv8 = self.conv_conv_identity_pool_crop(up8, [sfn*2, sfn*2], self.trainable, name='up8', pool=False,
                                                  padding='valid')
        up9 = self.crop_upsample_concat(conv8, conv1, 176, name='9')
        conv9 = self.conv_conv_identity_pool_crop(up9, [sfn, sfn], self.trainable, name='up9', pool=False,
                                                  padding='valid')

        self.pred = tf.layers.conv2d(conv9, class_num, (1, 1), name='final', activation=None, padding='same')
        self.output = tf.nn.softmax(self.pred)
