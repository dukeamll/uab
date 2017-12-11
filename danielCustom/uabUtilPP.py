#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 19:44:08 2017

@author: Daniel

functions to do post-processing on the network output (binary maps)
"""

import scipy.misc
import os
from skimage import measure
import numpy as np

def uabPPremSmClose(dirPath, tileNames, tr):
    #post-processing function that removes small regions in the output maps and then does morphological closing
    #dirPath -> path to the results of the network
    #tileNames -> name of tiles (not full path)
    #tr -> threshold for regions to remove
    #
    #returns a path
    
    tileInd = 0
    resPathPP = os.path.join(dirPath,'PP','tr' + str(tr))
    for tile in tileNames:
        imOut = scipy.misc.imread(os.path.join(dirPath, tile + 'preds.png'))    
        imOut[imOut != 0] = 1
        
        concomp, maxNum = measure.label(imOut,connectivity=1,return_num=True)
        all_labels = concomp.flatten()
        N,_ = scipy.histogram(all_labels, bins=range(0,maxNum+1,1))
        NabTr = np.where(N > tr)
        
        rr = np.zeros(all_labels.shape)
        for ind in NabTr[0][1:]:
            rr[all_labels == ind] = 1
        
        rr = np.reshape(rr,(imOut.shape))
        
        imClose = scipy.ndimage.morphology.binary_closing(rr)
        imClose = imClose.astype(np.uint8)
        imClose[imClose != 0] = 1

        if not resPathPP or not os.path.isdir(resPathPP):
            os.makedirs(resPathPP)
            
        scipy.misc.imsave(os.path.join(resPathPP,tile+'preds.png'), imClose)
        tileInd+=1
        if(tileInd % 10 == 0):
            print "Finished tile " + str(tileInd)

    return resPathPP