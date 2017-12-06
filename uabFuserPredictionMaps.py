#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 20:25:25 2017

@author: Daniel

base class for prediction fusion.  ALso remaps the output
Default class performs argmax on the predictions.  If you want something else, overload combineMapFunction
"""

import numpy as np

class uabFusePredictionMaps(object):
    def __init__(self, labelMapping = {0:0, 1:255}):
        self.name = 'argmax'
        self.labelMapping = labelMapping 
    
    def combineMaps(self, pred):
        output = self.combineMapFunction(pred)
        encode_func = np.vectorize(lambda x, y: y[x])
        return encode_func(output, self.labelMapping)
        
    def combineMapFunction(self, pred):
        if len(pred.shape) == 4:
            n, h, w, c = pred.shape
            outputs = np.zeros((n, h, w, 1), dtype=np.uint8)
            for i in range(n):
                outputs[i] = np.expand_dims(np.argmax(pred[i,:,:,:], axis=2), axis=2)
            return outputs
        elif len(pred.shape) == 3:
            outputs = np.argmax(pred, axis=2)
            return outputs
        
    def combineMapFunction_soft(self, pred):
        if len(pred.shape) == 4:
            n, h, w, c = pred.shape
            outputs = np.zeros((n, h, w, 1), dtype=np.uint8)
            for i in range(n):
                outputs[i] = np.expand_dims(pred[i,:,:,0], axis=2)
            return outputs
        elif len(pred.shape) == 3:
            outputs = pred[:, :, 0]
            return outputs

class uabFusePredictionAndAugs(uabFusePredictionMaps):
    def __init__(self, labelMapping = {0:0, 1:255}, combFun=1):
        self.combFun = combFun
        super(uabFusePredictionAndAugs, self).__init__(labelMapping= labelMapping)
        if(self.combFun == 1):
            self.name = 'AugMax'
        else:
            self.name = 'AugNOR'
        
    def combineMapFunction(self, pred):
        
        if(self.combFun == 1):
            #max 
            temp = np.max(pred, axis=3)
        elif(self.combFun == 2):
            #noisy-or
            temp = 1-np.prod(1-pred, axis=3)
                    
        return np.argmax(temp, axis=2)