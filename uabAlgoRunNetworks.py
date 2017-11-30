#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 15:50:00 2017

@author: Daniel
Class for training and testing networks

Training procedure

Testing procedure
(1) load in the fine-tuned unet 
(2) point it to the UM testing dataset
(3) Generate confidence maps
(4) convert maps to a RLE file to submit to the competition
"""
from __future__ import division
import os, time, shutil
import numpy as np
import tensorflow as tf
import scipy.misc
import uabUtilSubm
from uabDataReader import ImageLabelReader
import uabFuserPredictionMaps


class uabAlgorithmRunNetwork(object):
    
    ckptName = 'checkPoint'
    
    def __init__(self, netw, nEpochs = 15, batchSize = 10, n_train=8000, RANDOM_SEED = 1234, dec_rate=0.1, dec_step=10, learning_rate=1e-4, gpuDev = '0', predictionMapFuser = uabFuserPredictionMaps.uabFusePredictionMaps()):
        # environment settings        
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = gpuDev
        np.random.seed(RANDOM_SEED)
        tf.set_random_seed(RANDOM_SEED)
        
        self.network = netw
        self.combinePredictionMaps = predictionMapFuser
            
        self.modDir = uabUtilSubm.getBlockDir('network', self.network.makeName())
        self.nEpochs = nEpochs
        self.batchSize = batchSize
        self.nTrain = n_train
        self.global_step = []
        self.global_step_value = 0
        self.decayRate = dec_rate
        self.decStep = dec_step
        self.learningRate = learning_rate
        
        self.summary = []
        self.optimizer = []
        self.update_ops = None
        
    
    def makeCheckpointDir(self):
        #make directory for the checkpoint
        pathName = os.path.join(self.modDir, uabAlgorithmRunNetwork.ckptName)
        if(not(os.path.isdir(pathName))):
            os.makedirs(pathName)   
            
        return pathName
    
    def initRun(self):
        #set up the run parameters
        self.global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name='global_step')
        self.network.initGraph()
        self.network.make_loss('y')
        self.make_learning_rate(self.learningRate,
                             tf.cast(self.nTrain/self.batchSize * self.decStep, tf.int32), self.decayRate)
        self.make_update_ops('X', 'y')
        self.make_optimizer(self.learning_rate)
        self.make_summary()
    
    
    
    def make_learning_rate(self, lr, decay_steps, decay_rate):
        self.learning_rate = tf.train.exponential_decay(lr, self.global_step, decay_steps,
                                                        decay_rate, staircase=True)
    def make_update_ops(self, x_name, y_name):
        tf.add_to_collection('inputs', self.network.inputs[x_name])
        tf.add_to_collection('inputs', self.network.inputs[y_name])
        tf.add_to_collection('outputs', self.network.pred)
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        
    def make_optimizer(self, lr):
        with tf.control_dependencies(self.update_ops):
            self.optimizer = tf.train.AdamOptimizer(lr).minimize(self.network.loss, global_step=self.global_step)

    def make_summary(self):
        tf.summary.histogram('Predicted_Prob', tf.argmax(tf.nn.softmax(self.network.pred), 1))
        tf.summary.scalar('Cross_Entropy', self.network.loss)
        tf.summary.scalar('learning_rate', self.learning_rate)
        self.summary = tf.summary.merge_all()
    

    def trainCNNmodel(self, colObj, extrObj, cities='', TrTiles='', ValTiles='', verb_step=100, forceRun = 0, resumeTrain = 0,trAug=''):
        #function to train a network.  assumes you've got a graph defined, images made
        #first checks if the checkpoint directory exists for this model, otherwise trains
        #resumeTrain is set if you want more epochs or something otherwise, checks if you wanted to pretrain otherwise goes from scratch.  Forcerun is if you want to overwrite results
        #
        #cities and tiles are set in order to subselect data for training/validation
        
        print 'Start CNN training'
        ckptName = self.makeCheckpointDir()
        
        if forceRun == 0 and os.listdir(ckptName) != "":
            return ckptName
        
        if(forceRun == 1):
            shutil.rmtree(ckptName)
        
        #initialize
        tf.reset_default_graph()
        
        #get the iterators
        _, trainIter = extrObj.makeDataReader(colObj, 1, self.batchSize, {'city':cities, 'tileInds':TrTiles},dataAug=trAug)
        _, validIter = extrObj.makeDataReader(colObj, 1, self.batchSize, {'city':cities, 'tileInds':ValTiles})
        
        #set up graph
        self.initRun()
        
        #pretrain or continue training
        if resumeTrain == 1:
            #continue training if possible
            startPointDir = ckptName
        elif resumeTrain == 0 and self.network.ckptDir:
            #if you don't want to force_run but the network has a checkpoint directory, then use it for pretraining
            startPointDir = self.network.ckptDir
        else:
            startPointDir = []
        
        config = tf.ConfigProto()
        
        #tensorflow necessities
        start_time = time.time()
        with tf.Session(config=config) as sess:
            #init = tf.global_variables_initializer()
            #sess.run(init)
    
            saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)
    
            #if you want to continue training
            if startPointDir and os.path.exists(startPointDir) and tf.train.get_checkpoint_state(startPointDir):
                latest_check_point = tf.train.latest_checkpoint(startPointDir)
                saver.restore(sess, latest_check_point)
                print 'Pretrain from ' + startPointDir
    
            threads = tf.train.start_queue_runners(coord=extrObj.coord, sess=sess)
            try:
                train_summary_writer = tf.summary.FileWriter(ckptName, sess.graph)                
                # define summary operations
                valid_cross_entropy = tf.placeholder(tf.float32, [])
                valid_cross_entropy_summary_op = tf.summary.scalar('xent_validation', valid_cross_entropy)
                
                ######### Training begins #############
                
                for epoch in range(self.nEpochs):
                    for step in range(0, self.nTrain, self.batchSize):
                        #training
                        _, self.global_step_value, X_batch, y_batch = self.network.trainNetworkOp(trainIter, sess, self.global_step, self.optimizer, True)
                        
                        #output info about training
                        if self.global_step_value % verb_step == 0:
                            pred_train, step_cross_entropy, step_summary = sess.run([self.network.pred, self.network.loss, self.summary], feed_dict={self.network.inputs['X']: X_batch, self.network.inputs['y']: y_batch, self.network.inputs['mode']: True})
                            train_summary_writer.add_summary(step_summary, self.global_step_value)
                        
                            print('Epoch {:d} step {:d}\tcross entropy = {:.3f}'.
                                  format(epoch, self.global_step_value, step_cross_entropy))
                    
                    #operations about validation
                    print 'Validation Start'
                    pred_valid, cross_entropy_valid, _, _ = self.network.trainNetworkOp(validIter, sess, self.global_step, self.optimizer, False)
                    print('Validation cross entropy: {:.3f}'.format(cross_entropy_valid))
                    valid_cross_entropy_summary = sess.run(valid_cross_entropy_summary_op,
                                                       feed_dict={valid_cross_entropy: cross_entropy_valid})
                    train_summary_writer.add_summary(valid_cross_entropy_summary, self.global_step_value)
                
            finally:
                extrObj.coord.request_stop()
                extrObj.coord.join(threads)
                saver.save(sess, '{}/model.ckpt'.format(ckptName), global_step=self.global_step)
            
        duration = time.time() - start_time
        print('duration {:.2f} hours'.format(duration/60/60))
        
        return ckptName
    
    def testCNNmodel(self, colObj, imFiles, ckptDir, INPUT_SIZE = np.array((0, 0)), forceRun = 0, isVal=0, savePredictions=0):
        #function to run a saved CNN model.  This is expected to be a fixed process and therefore it is not being more systematized than this
        #takes in a model name, a checkpoint directory, a list of files to test on
        #
        #it will typically always output files in the same place (outputLabels) but you can override that if necessary
        #
        #if input_size is different from zero, then use that size, otherwise just use the size of the tiles as the input
        
        
        print 'Start CNN testing'
        # image reader
        tf.reset_default_graph()
        coord = tf.train.Coordinator()
        config = tf.ConfigProto()
        
        #precompute the amount of padding required
        if(INPUT_SIZE is not np.array((0,0))):
            INPUT_SIZE = colObj.tileSize[:2]
        
        INPUT_SIZE = np.array(INPUT_SIZE)
        valsz = self.network.getNextValidInputSize(INPUT_SIZE + self.network.getRequiredPadding())
        padAmt = (valsz - INPUT_SIZE)/2
        padAmt = padAmt.astype(np.int)
        
        self.network.inpSize = INPUT_SIZE + 2*padAmt
        self.network.initGraph(toLoad=0)
        #make results folder
        resPath = self.modDir
        
        resFls = os.listdir(resPath)
        if(len(resFls) > 0 and forceRun == 0):
            return resPath
        
        #collection extensions
        #exsts = colObj.getDataExtensions()
        exsts = ['data', 'dif']
        
        
        start_time = time.time()
        with tf.Session(config=config) as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
        
            saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)
        
            if os.path.exists(ckptDir) and tf.train.get_checkpoint_state(ckptDir):
                latest_check_point = tf.train.latest_checkpoint(ckptDir)
                saver.restore(sess, latest_check_point)
                print 'Load model from ' + latest_check_point
        
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)
            try:
                for image_name in imFiles:
                    # load reader
                    #get all the datafiles that this network runs on 
                    impath = [colObj.getDataNameByTile(image_name, ext) for ext in exsts]

                    
                    iterator_test = ImageLabelReader.getTestIterator(
                        impath,
                        batch_size=1,
                        tile_dim=INPUT_SIZE,
                        patch_size=self.network.inpSize,
                        overlap=0, padding=padAmt)
                    
                    # run
                    result = self.network.testNetworkOp(sess, iterator_test)
                    
                    #this may result in there being a larger output patch because of the padding.  Assumes that the center is the valid region & that it is symmetric otherwise error
                    rShap = result.shape
                    bdPad = (np.array(rShap[1:3]) - colObj.tileSize)/2
                    if (bdPad % 1 != 0).any():
                        raise NotImplementedError('The offset is not symmetric.  This is not handled')
                    bdPad = bdPad.astype(np.int)
                    image_pred = result[0,bdPad[0]:-bdPad[0],bdPad[1]:-bdPad[1],:]
                    
                    combPreds = self.combinePredictionMaps.combineMaps(image_pred)
                    
                    """
                    result = utils.get_output_label(result,
                                                  result.shape,
                                                  INPUT_SIZE,
                                                  {0:0, 1:255}, overlap=0,
                                                  output_image_dim=INPUT_SIZE,
                                                  output_patch_size=INPUT_SIZE)
                    """
                    u = image_name.split('/')
                    u1 = u[-1].split('_RGB.tif')
                    
                    if(isVal == 1):
                        savePath = os.path.join(resPath,'val')
                        if not savePath:
                            os.makedirs(savePath)
                    else:
                        savePath = resPath
                    
                    scipy.misc.imsave(os.path.join(savePath,u1[0]+'preds.png'), combPreds)
                    if(savePredictions):
                        scipy.misc.imsave(os.path.join(savePath,u1[0]+'cfs.png'), image_pred[:,:,1])
                    
                    
                    #get the image name & add something to it to show its the GT
                    '''u = image_name.split('/')
                    u1 = u[-1].split('_RGB.tif')
                    scipy.misc.imsave(os.path.join(resPath,u1[0]+'preds.png'), np.argmax(result[0,:,:,:],axis=2))'''
                    
            finally:
                coord.request_stop()
                coord.join(threads)
        
        duration = time.time() - start_time
        print('duration {:.2f} minutes'.format(duration/60))
        return resPath