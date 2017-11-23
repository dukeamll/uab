#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 19:02:58 2017

@author: Daniel

Submission script
"""

import uab_collectionFunctions
from uabMakeNetwork import uabNetUnetDeflt
from uabAlgoRunNetworks import uabAlgorithmRunNetwork
import uabUtilSubm
import tensorflow as tf
import os


tf.reset_default_graph()
colN = 'USSOCOM-BLDC'
blCol = uab_collectionFunctions.uabCollection(colN, '_custGT.png', {'data':'_RGB.tif'})

ckptDir = './Models/UNET_um_no_random_resampled_7'
INPSIZE=(2048,2048)
    
#make a network
unetMo = uabNetUnetDeflt(input_size=INPSIZE)
networkRouter = uabAlgorithmRunNetwork(unetMo)

#run the network
imFiles = [os.path.join(blCol.imDirectory, 'testing', a) for a in blCol.tileTeList]
resPath = networkRouter.testCNNmodel(blCol, imFiles, ckptDir, INPUT_SIZE=INPSIZE, forceRun=0)

textPath = uabUtilSubm.makeSubmissionFile(blCol, resPath, networkRouter.network.getName(), 'submUM_PR7.txt')