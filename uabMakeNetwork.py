#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 15:28:16 2017

@author: Daniel
classes to handle network making.  This is where you'll make the custom architectures etc
"""

from __future__ import division
import tensorflow as tf
import numpy as np

class uabNetArchis(object):
    
    resDir = 'network'
    def __init__(self, custName, input_size, nClasses=2, ndims=3, pretrainDict = {}):
        self.inpSize = input_size
        self.featDims = ndims
        self.nClasses = nClasses
        self.pred = []
        self.inputs = {}
        self.model_name = custName
        
        self.ckptDir = []
        self.pretDict = pretrainDict
        self.loss = []
        
        #load the network graph
        self.initGraph(toLoad=0)
            
        self.makeName()
    
    def initGraph(self, toLoad=1):
        #initialize the graph & make the sizes etc.
        # define place holders
        X = tf.placeholder(tf.float32, shape=[None, self.inpSize[0], self.inpSize[1], self.featDims], name='X')
        y = tf.placeholder(tf.int32, shape=[None, self.inpSize[0], self.inpSize[1], 1], name='y')
        mode = tf.placeholder(tf.bool, name='mode')
        self.inputs = {'X': X, 'y': y,'mode':mode}
        
        #actually do the graph creation thing
        self.makeGraph(X, y, mode)
        
        #do pretraining if a custom directory is specified
        if self.pretDict:
            if 'name' in self.pretDict:
                self.isPretrained = self.pretDict['name']
            else:
                self.isPretrained = 1
            if(toLoad == 1):
                self.load_weights(self.pretDict)
        else:
            self.isPretrained = 0
    
    def makeGraph(self, X, y, mode):
        #function to be implemented in the subclass.  defines the network architecture
        raise NotImplementedError('Must be implemented by the subclass')
    
    def getNextValidInputSize(self, sz):
        #some network (e.g., cropping unet) can only accept inputs of particular sizes.  This function returns the next valid size.  Default is to return the same size (which assumes that any size is fine)
        return sz
    
    def getRequiredPadding(self):
        #based on the architecture, some cropping might occur and therefore we need to know how much padding may be required.  Default is 0
        return 0
    
    def makeName(self):
        #naming function based on the network parameters
        if(self.isPretrained):
            if(self.isPretrained == 1):
                nm = '_isPT'
            else:
                nm = '_'+self.isPretrained
        else:
            nm = ''
        
        compName = self.getName()
        return compName + nm
        
    def getName(self):
        raise NotImplementedError('Must be implemented by the subclass')
    
    def make_loss(self, y_name):
        raise NotImplementedError('Must be implemented by the subclass')
    
    def load_weights(self, pretDict):
        #this is an example of loading pretrained weights into the network
        #The input to the function is a dictionary, that, at a minimum, has the directory from which to load the model weights
        #
        #Note: this operation depends on the names of the layers and needs to be matched up to the architecture you are using.  For this reason, this function may need to be reimplemented for your particular usage
        
        self.ckptDir = pretDict['ckpt_dir']
        layers2load = pretDict['layers2load']
        
        layers_list = []
        for layer_id in layers2load.split(','):
            layer_id = int(layer_id)
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
        
        tf.contrib.framework.init_from_checkpoint(self.ckptDir, load_dict)
        
        
    def trainNetworkOp(self, data_iter, sess, globStep, optm, mode, data_reader = []):
        #convenience function to pass training data through the network

        if data_iter is not None:
            X_batch, y_batch = sess.run(data_iter)
        else:
            X_batch, y_batch = next(data_reader)
            
        a, b = sess.run([optm, globStep], 
                        feed_dict={self.inputs['X']:X_batch,
                        self.inputs['y']:y_batch,
                        self.inputs['mode']: mode})
    
        return a, b, X_batch, y_batch
    
    def testNetworkOp(self, sess, test_iterator):
        #convenience function to feed testing data to a network
        result = []
        for X_batch in test_iterator:
            pred = sess.run(self.pred, feed_dict={self.inputs['X']:X_batch,
                                                  self.inputs['mode']: False})
            result.append(pred)
        result = np.vstack(result)
        return result
    
    def testNetworkOp_softmax(self, sess, test_iterator):
        #convenience function to feed testing data to a network
        result = []
        for X_batch in test_iterator:
            pred = sess.run(tf.nn.softmax(self.pred), 
                            feed_dict={self.inputs['X']:X_batch,
                                       self.inputs['mode']: False})
            result.append(pred)
        result = np.vstack(result)
        return result

##################################    
## Useful functions for networks        
##################################
    def conv_conv_pool(self, input_, n_filters, training, name, conv_strid=(3, 3),
                       pool=True, pool_size=(2, 2), pool_stride=(2, 2),
                       activation=tf.nn.relu, padding='same', bn=True):
        net = input_

        with tf.variable_scope('layer{}'.format(name)):
            for i, F in enumerate(n_filters):
                net = tf.layers.conv2d(net, F, conv_strid, activation=None,
                                       padding=padding, name='conv_{}'.format(i + 1))
                if bn:
                    net = tf.layers.batch_normalization(net, training=training, name='bn_{}'.format(i+1))
                net = activation(net, name='relu_{}'.format(name, i + 1))

            if pool is False:
                return net

            pool = tf.layers.max_pooling2d(net, pool_size, strides=pool_stride, name='pool_{}'.format(name))
            return net, pool

    def concat(self, input_a, input_b, training, name):
        with tf.variable_scope('layer{}'.format(name)):
            inputA_norm = tf.layers.batch_normalization(input_a, training=training, name='bn')
            return tf.concat([inputA_norm, input_b], axis=-1, name='concat_{}'.format(name))

    def upsampling_2D(self, tensor, name, size=(2, 2)):
        H, W, _ = tensor.get_shape().as_list()[1:]  # first dim is batch num
        H_multi, W_multi = size
        target_H = H * H_multi
        target_W = W * W_multi

        return tf.image.resize_nearest_neighbor(tensor, (target_H, target_W), name='upsample_{}'.format(name))

    def upsample_concat(self, input_a, input_b, name):
        upsample = self.upsampling_2D(input_a, size=(2, 2), name=name)
        return tf.concat([upsample, input_b], axis=-1, name='concat_{}'.format(name))
    
    def crop_upsample_concat(self, input_a, input_b, margin, name):
        with tf.variable_scope('crop_upsample_concat'):
            _, w, h, _ = input_b.get_shape().as_list()
            input_b_crop = tf.image.resize_image_with_crop_or_pad(input_b, w-margin, h-margin)
            return self.upsample_concat(input_a, input_b_crop, name)

    def fc_fc(self, input_, n_filters, training, name, activation=tf.nn.relu, dropout=True):
        net = input_
        with tf.variable_scope('layer{}'.format(name)):
            for i, F in enumerate(n_filters):
                net = tf.layers.dense(net, F, activation=None)
                if activation is not None:
                    net = activation(net, name='relu_{}'.format(name, i + 1))
                if dropout:
                    net = tf.layers.dropout(net, rate=self.dropout_rate, training=training, name='drop_{}'.format(name, i + 1))
        return net
    


#this class basically creates Bohao's unet 
class uabNetUnetDeflt(uabNetArchis):
    def __init__(self, input_size, model_name='', nClasses=2, start_filter_num=32, ndims=3, pretrainDict = {}):
        self.snf = start_filter_num
        self.model_name = 'defUnet'
        super(uabNetUnetDeflt, self).__init__(self.model_name + model_name, input_size, nClasses=nClasses, ndims=ndims, pretrainDict = pretrainDict)
        
    def makeGraph(self, X, y, mode):
        sfn = self.snf
        # downsample
        conv1, pool1 = self.conv_conv_pool(X, [sfn, sfn], mode, name='conv1')
        conv2, pool2 = self.conv_conv_pool(pool1, [sfn*2, sfn*2], mode, name='conv2')
        conv3, pool3 = self.conv_conv_pool(pool2, [sfn*4, sfn*4], mode, name='conv3')
        conv4, pool4 = self.conv_conv_pool(pool3, [sfn*8, sfn*8], mode, name='conv4')
        conv5 = self.conv_conv_pool(pool4, [sfn*16, sfn*16], mode, name='conv5', pool=False)

        # upsample
        up6 = self.upsample_concat(conv5, conv4, name='6')
        conv6 = self.conv_conv_pool(up6, [sfn*8, sfn*8], mode, name='up6', pool=False)
        up7 = self.upsample_concat(conv6, conv3, name='7')
        conv7 = self.conv_conv_pool(up7, [sfn*4, sfn*4], mode, name='up7', pool=False)
        up8 = self.upsample_concat(conv7, conv2, name='8')
        conv8 = self.conv_conv_pool(up8, [sfn*2, sfn*2], mode, name='up8', pool=False)
        up9 = self.upsample_concat(conv8, conv1, name='9')
        conv9 = self.conv_conv_pool(up9, [sfn, sfn], mode, name='up9', pool=False)

        self.pred = tf.layers.conv2d(conv9, self.nClasses, (1, 1), name='final', activation=None, padding='same')
        
    def getName(self):
        #naming here is mostly for backward compatibility
        return self.model_name
    
    def make_loss(self, y_name):
        with tf.variable_scope('loss'):
            pred_flat = tf.reshape(tf.nn.softmax(self.pred), [-1, self.nClasses])
            y_flat = tf.reshape(tf.squeeze(self.inputs[y_name], axis=[3]), [-1, ])
            indices = tf.squeeze(tf.where(tf.less_equal(y_flat, self.network.nClasses - 1)), 1)
            gt = tf.gather(y_flat, indices)
            prediction = tf.gather(pred_flat, indices)
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt))



class uabNetUnetCrop(uabNetArchis):
    def __init__(self, input_size, model_name='', nClasses=2, start_filter_num=32, ndims=3, pretrainDict = {}):
        self.snf = start_filter_num
        self.model_name = 'cropUnet'
        super(uabNetUnetCrop, self).__init__(self.model_name + model_name, input_size, nClasses=nClasses, ndims=ndims, pretrainDict = pretrainDict)           
        
    def makeGraph(self, X, y, mode):
        sfn = self.snf

        # downsample
        conv1, pool1 = self.conv_conv_pool(X, [sfn, sfn], mode, name='conv1', padding='valid')
        conv2, pool2 = self.conv_conv_pool(pool1, [sfn*2, sfn*2], mode, name='conv2', padding='valid')
        conv3, pool3 = self.conv_conv_pool(pool2, [sfn*4, sfn*4], mode, name='conv3', padding='valid')
        conv4, pool4 = self.conv_conv_pool(pool3, [sfn*8, sfn*8], mode, name='conv4', padding='valid')
        conv5 = self.conv_conv_pool(pool4, [sfn*16, sfn*16], mode, name='conv5', pool=False, padding='valid')

        # upsample
        up6 = self.crop_upsample_concat(conv5, conv4, 8, name='6')
        conv6 = self.conv_conv_pool(up6, [sfn*8, sfn*8], mode, name='up6', pool=False, padding='valid')
        up7 = self.crop_upsample_concat(conv6, conv3, 32, name='7')
        conv7 = self.conv_conv_pool(up7, [sfn*4, sfn*4], mode, name='up7', pool=False, padding='valid')
        up8 = self.crop_upsample_concat(conv7, conv2, 80, name='8')
        conv8 = self.conv_conv_pool(up8, [sfn*2, sfn*2], mode, name='up8', pool=False, padding='valid')
        up9 = self.crop_upsample_concat(conv8, conv1, 176, name='9')
        conv9 = self.conv_conv_pool(up9, [sfn, sfn], mode, name='up9', pool=False, padding='valid')

        self.pred = tf.layers.conv2d(conv9, self.nClasses, (1, 1), name='final', activation=None, padding='same')
        
    def getName(self):
        return self.model_name + '_sfn%d' % (self.snf)
    
    def make_loss(self, y_name):
        with tf.variable_scope('loss'):
            pred_flat = tf.reshape(tf.nn.softmax(self.pred), [-1, self.nClasses])
            _, w, h, _ = self.inputs[y_name].get_shape().as_list()
            y = tf.image.resize_image_with_crop_or_pad(self.inputs[y_name], w-184, h-184)
            y_flat = tf.reshape(tf.squeeze(y, axis=[3]), [-1, ])
            indices = tf.squeeze(tf.where(tf.less_equal(y_flat, self.nClasses - 1)), 1)
            gt = tf.gather(y_flat, indices)
            prediction = tf.gather(pred_flat, indices)
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt))
    
    def getNextValidInputSize(self, sz):
        return 16*(np.ceil((sz - 124)/16))+124
    
    def getRequiredPadding(self):
        return 92*2
    
class uabNetUnetCrop_Appendix(uabNetArchis):
    def __init__(self, input_size, model_name='', nClasses=2, start_filter_num=32, ndims=3, pretrainDict = {}):
        self.snf = start_filter_num
        self.model_name = 'cropUnet_Appendix_'
        super(uabNetUnetCrop_Appendix, self).__init__(self.model_name + model_name, input_size, nClasses=nClasses, ndims=ndims, pretrainDict = pretrainDict)           
        
    def makeGraph(self, X, y, mode):
        sfn = self.snf

        # downsample
        conv1, pool1 = self.conv_conv_pool(X, [sfn, sfn], mode, name='conv1', padding='valid')
        conv2, pool2 = self.conv_conv_pool(pool1, [sfn*2, sfn*2], mode, name='conv2', padding='valid')
        conv3, pool3 = self.conv_conv_pool(pool2, [sfn*4, sfn*4], mode, name='conv3', padding='valid')
        conv4, pool4 = self.conv_conv_pool(pool3, [sfn*8, sfn*8], mode, name='conv4', padding='valid')
        conv5 = self.conv_conv_pool(pool4, [sfn*16, sfn*16], mode, name='conv5', pool=False, padding='valid')

        # upsample
        up6 = self.crop_upsample_concat(conv5, conv4, 8, name='6')
        conv6 = self.conv_conv_pool(up6, [sfn*8, sfn*8], mode, name='up6', pool=False, padding='valid')
        up7 = self.crop_upsample_concat(conv6, conv3, 32, name='7')
        conv7 = self.conv_conv_pool(up7, [sfn*4, sfn*4], mode, name='up7', pool=False, padding='valid')
        up8 = self.crop_upsample_concat(conv7, conv2, 80, name='8')
        conv8 = self.conv_conv_pool(up8, [sfn*2, sfn*2], mode, name='up8', pool=False, padding='valid')
        up9 = self.crop_upsample_concat(conv8, conv1, 176, name='9')
        conv9 = self.conv_conv_pool(up9, [sfn, sfn], mode, name='up9', pool=False, padding='valid')
        
        conv10 = tf.layers.conv2d(conv9, sfn, (1, 1), name='second_final', padding='same')
        self.pred = tf.layers.conv2d(conv10, self.nClasses, (1, 1), name='final', activation=None, padding='same')
        
    def getName(self):
        return self.model_name + '_sfn%d' % (self.snf)
    
    def make_loss(self, y_name):
        with tf.variable_scope('loss'):
            pred_flat = tf.reshape(tf.nn.softmax(self.pred), [-1, self.nClasses])
            _, w, h, _ = self.inputs[y_name].get_shape().as_list()
            y = tf.image.resize_image_with_crop_or_pad(self.inputs[y_name], w-184, h-184)
            y_flat = tf.reshape(tf.squeeze(y, axis=[3]), [-1, ])
            indices = tf.squeeze(tf.where(tf.less_equal(y_flat, self.nClasses - 1)), 1)
            gt = tf.gather(y_flat, indices)
            prediction = tf.gather(pred_flat, indices)
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt))
    
    def getNextValidInputSize(self, sz):
        return 16*(np.ceil((sz - 124)/16))+124
    
    def getRequiredPadding(self):
        return 92*2