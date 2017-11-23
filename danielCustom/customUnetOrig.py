#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 23:20:18 2017

@author: Daniel

class to house the unet with custom layers
"""

from uabRepoCode.uabMakeNetwork import uabNetArchis
import tensorflow as tf

class uabNetUnetCropClLays(uabNetArchis):
    def __init__(self, input_size, finDecLay = 9, numClLay = 0, startNumClFilt = 100, model_name='', nClasses=2, start_filter_num=32, ndims=3, pretrainDict = {}):
        self.snf = start_filter_num
        self.model_name = 'UnetCropCL'
        self.finDecoderLay = finDecLay
        self.nClassifLayers = numClLay
        self.nStartClFilt = startNumClFilt
        super(uabNetUnetCropClLays, self).__init__(self.model_name + model_name, input_size, nClasses=nClasses, ndims=ndims, pretrainDict = pretrainDict)           
        
    def makeGraph(self, X, y, mode):
        sfn = self.snf

        # downsample
        conv1, pool1 = self.conv_conv_pool(X, [sfn, sfn], mode, name='conv1', padding='valid')
        conv2, pool2 = self.conv_conv_pool(pool1, [sfn*2, sfn*2], mode, name='conv2', padding='valid')
        conv3, pool3 = self.conv_conv_pool(pool2, [sfn*4, sfn*4], mode, name='conv3', padding='valid')
        conv4, pool4 = self.conv_conv_pool(pool3, [sfn*8, sfn*8], mode, name='conv4', padding='valid')
        conv5 = self.conv_conv_pool(pool4, [sfn*16, sfn*16], mode, name='conv5', pool=False, padding='valid')

        # upsample -- decoder layers
        # how many of these to keep
        vals = [[conv4, 8, 8], 
                [conv3, 32, 4],
                [conv2, 80, 2],
                [conv1, 176, 1]]
        ind = 6
        convn = []
        upL = []
        nn = 0
        for vv in vals:
            if(ind == 6):
                convn.append(conv5)
            
            if(self.finDecoderLay > ind - 1):    
                upL.append(self.crop_upsample_concat(convn[nn], vv[0], vv[1], name=str(ind)))
                convn.append(self.conv_conv_pool(upL[nn], [sfn*vv[2], sfn*vv[2]], mode, name='up'+str(ind), pool=False, padding='valid'))
                ind += 1
                nn += 1
            else:
                break
        """
        #classification layers -- how many to add.  These should be fully convolutional 1x1xN
        nLays = [self.nStartClFilt*2**(index-1) for index in range(self.nClassifLayers)]
        nLays.append(self.nClasses)
        
        conv_last = tf.stop_gradient(convn[-1])
        ppLay = []
        ppLay.append(conv_last)
        nn = 0
        for i in range(len(nLays)):
            if(i == len(nLays)-1):
                nm = 'Ffinal'
            else:
                nm = 'clLay' + str(i+1)
            
            input_ = tf.layers.conv2d(ppLay[nn], nLays[i], (1, 1), name=nm, activation=None, padding='same')
            net = tf.layers.batch_normalization(input_, training=mode, name='bnCl_{}'.format(i+1))
            net = tf.nn.relu(net, name='reluCl_{}'.format(nm, i + 1))
            
            ppLay.append(net)
            nn += 1
            self.pred = ppLay[-1]
        """
        conv_last = tf.stop_gradient(convn[-1])
        shpVar = tf.get_variable("shVar",[1,1,conv_last.shape[-1],2])
        self.pred = tf.nn.conv2d(conv_last, shpVar, (1,1,1,1), name='final', padding='SAME')
        
    def getName(self):
        return self.model_name + '_nDc%d_sfn%d_nCl%d_sCln%d' % ( self.finDecoderLay, self.snf, self.nClassifLayers+1, self.nStartClFilt)
    
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
