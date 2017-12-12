#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 15:01:14 2017

@author: jordan

Classes defined to split the dataset for training, validation, and testing.  The folds are one number per chip or tile (depending on the input)
"""

import os

def uabUtilGetFolds(parentDir, fileList, xvalType):
    if(xvalType == 0):
        xvalObj = uabXvalByCity()
    
    return xvalObj.getFolds(parentDir, fileList)
    

class uabXvalParent(object):
    def getFolds(self, parentDir, fileList):
        if(isinstance(fileList, str)):
            filename = os.path.join(parentDir,fileList) 
            with open(filename) as file:
                chipFiles = file.readlines()
            
            chipFiles = [a.strip().split(' ') for a in chipFiles]
        else:
            chipFiles = fileList
            
        return self.computeFolds(chipFiles)
    
    def computeFolds(self, chipFiles):
        #one index per row in the list of lists of chips or tiles by their channel
        raise NotImplementedError('Must be implemented by the subclass')

class uabXvalByCity(uabXvalParent):
    def computeFolds(self, chipFiles):
        pass