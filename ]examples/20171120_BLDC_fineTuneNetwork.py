#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 13:32:27 2017

@author: Daniel

script that does everything from start to finish
(1) load in collection lists
(2) extract patches 
(3) make the training & validation readers
(4) make a network
(5) train network
(6) test network
(7) make submission
"""

import uab_collectionFunctions
import uab_DataHandlerFunctions 
from uabMakeNetwork import uabNetUnetCrop
from uabAlgoRunNetworks import uabAlgorithmRunNetwork
import uabUtilSubm, os
import tensorflow as tf

tf.reset_default_graph()
#(1) load in collection lists
colN = 'USSOCOM-BLDC'
CITY_NAME = 'JAX,TAM'
#TRAIN_TILE_NAMES = ','.join(['{}'.format(i) for i in range(20,143)])
#VAL_TILE_NAMES = ','.join(['{}'.format(i) for i in range(0,20)])
TRAIN_TILE_NAMES = ','.join(['{}'.format(i) for i in range(5,20)])
VAL_TILE_NAMES = ','.join(['{}'.format(i) for i in range(0,5)])
BATCH_SIZE = 1

#blCol = uab_collectionFunctions.uabCollection(colN, '_custGT.png', {'data':'_RGB.tif','dtm':'_DTM.tif', 'dsm':'_DSM.tif'})
blCol = uab_collectionFunctions.uabCollection(colN, '_custGT.png', {'data':'_RGB.tif'})
blCol.tileTrList = blCol.tileTrList[:20]

#(2) extract patches for training
#value has to be 16n + 124
trSize = (572,572)
extrObj = uab_DataHandlerFunctions.uabDataHandler(cSize=trSize)
directory = extrObj.extractChips(blCol, 1, numPixOverlap=0, forceRun=0)
#(3) make the readers for training & validation
#turns out that validaiton is being done on chips so we just need the readers to know which tiles belong to training or validation

#(4) make a network
ckptDir = './Models/UnetInria_Origin_aug_fr_resample'
pretDir = {'ckpt_dir':ckptDir, 'layers2load':'1,2,3,4,5,6,7','name':'PrIr7'}
unetMo = uabNetUnetCrop(input_size=extrObj.chipExtrSize,pretrainDict=pretDir)
networkRouter = uabAlgorithmRunNetwork(unetMo, batchSize=BATCH_SIZE, nEpochs=1, n_train=10)

#(5) make training
resPath = networkRouter.trainCNNmodel(blCol, extrObj, cities=CITY_NAME, TrTiles=TRAIN_TILE_NAMES, ValTiles=VAL_TILE_NAMES, forceRun=1,resumeTrain=0,verb_step=2)

#(6) run the network
imFiles = [os.path.join(blCol.imDirectory, 'testing', a) for a in blCol.tileTeList]
INPSIZE=(2044,2044)
resPathT = networkRouter.testCNNmodel(blCol, imFiles, resPath, INPUT_SIZE=INPSIZE)
#(7) make submission files
textPath = uabUtilSubm.makeSubmissionFile(blCol, resPathT, networkRouter.network.model_name, 'submIR1.txt')
