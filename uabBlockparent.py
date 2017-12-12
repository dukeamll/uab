#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 14:55:26 2017

@author: Daniel
Handles the naming and running of objects
(1) make the path for the result of that block
(2) check whether the outcome already exists
(3) keeps track of the channels on which a block operates.  Obtain this information from the meta-data file associated with the collection

NOTE: this class should never be instantiated

Methods you must implement
(1) initAction
    any actions to initialize a block (beyond setting its properties) are to be defined in this function.  Outputs nothing
(2) algoName
    use the object properties to define a fixed name for this block.  Outputs a string    
(3) runAction
    In this method, the action of the block happens [e.g., patch extraction].  Returns an absolute path     
    this also handles the saving of any output made
(4) makeDirectoryPaths
    make the folder strucutre for the particular kind of block.  Here is an example for patch-extraction.  Returns a valid path starting after the results path
    Results/PatchExt/[PatchExtName]/[CollectionName]/files_ext.tif...
"""
from . import uabRepoPaths
import os
from . import util_functions

#dictionary that holds the 
outputDirs = {'preproc':'TilePreproc', 'patchExt':'PatchExtr'}

class uabBlock(object):
    
    def __init__(self, runChannels, name):
        self.name = name  
        #channels on which to run this block.  Get the channel indexes from the meta-data file associated with the collection
        self.runChannels = runChannels
        
    def initAction(self):
        raise NotImplementedError('Must be implemented by the subclass')
    
    def runAction(self, runData):
        raise NotImplementedError('Must be implemented by the subclass')
        
    def algoName(self):
        raise NotImplementedError('Must be implemented by the subclass')
        
    def getDirectoryPaths(self, colObj):
        raise NotImplementedError('Must be implemented by the subclass')
    
    def getName(self):
        nm = self.algoName()
        return nm.replace('.','p')
    
    def run(self, colObj, forcerun=0):
        #Check if the result that exists is finished.  If you want to overwrite a result, set forcerun == 1
        
        #get the path from this block for the results
        postDirs = self.getDirectoryPaths(colObj)
        
        path = self.getBlockDir(postDirs)
        stateFile = os.path.join(path, 'state.txt')
        stateExist = os.path.exists(stateFile)
        
        if(forcerun == 1 or stateExist == 0):
            self.runAtomic(colObj, stateFile)
        else:
            with open(stateFile, 'r') as f:
                a = f.readlines()
                if(a[0].strip() != 'Finished'):
                    self.runAtomic(colObj, stateFile)
        
        return path
    
    def runAtomic(self, colObj, stateFile):
        #checking the state file and running the block
        print(('Start running %s' % self.getName()))
        with open(stateFile, 'w') as f:
            f.write('Incomplete\n')
        
        self.runAction(colObj)
        
        with open(stateFile, 'w') as f:
            f.write('Finished\n')
    
    
    
    @staticmethod
    def getBlockDir(postDirs):
        #directories after the path for the results
        dirN = os.path.join(uabRepoPaths.resPath, postDirs)
        util_functions.uabUtilMakeDirectoryName(dirN)
        return dirN