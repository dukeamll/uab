#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 14:44:26 2017

@author: Daniel
Class to pre-process tiles.  This operates on specified channels and the list of tiles.  Examples are subtraction of heightmaps or rescaling
User must implement:
    initAction()
    algoName()
    runTilePreproc()
        this is the running function and gets called by the parent class runAction()
    
    
Implementation notes:
(1) A double for-loop over channels and tiles is pushed to the child class because it is conceivable that processing would occur over several tiles or channels and therefore, we cannot build this convenience into the parent class.
"""

import os
import numpy as np
from . import uabBlockparent
from .uabBlockparent import uabBlock
from . import util_functions

class uabPreprocClass(uabBlock):
    
    def __init__(self, runChannels, name, extension, description):
        super(uabPreprocClass, self).__init__(runChannels, name)
        #extension to save this file with (e.g., _Resc.npy)
        self.ext = extension
        #Human readable description of this process (e.g., Rescaling of the intensities in the tile by multiplying and adding a bias)
        self.descr = description
    
    def blockMetaDescription(self):
        return "{}\t{}\t{}\n".format(self.ext, os.path.join(uabBlockparent.outputDirs['preproc'], self.algoName()), self.descr)
    
    def getDirectoryPaths(self, colObj):
        return os.path.join(colObj.imDirectory, colObj.dataDirnames['data'], uabBlockparent.outputDirs['preproc'], self.algoName())
    
    def runAction(self, colObj):
        #handles running the preprocessing operation on the tile & updating the collection meta-data with this tile information
        self.runTilePreproc(colObj)
        
        updStr = self.blockMetaDescription()
        colObj.setMetadataFile(updStr)
        
    def runTilePreproc(self, colObj):
        raise NotImplementedError('Must be implemented by the subclass')

#class to save one channel from a multi-channel tile
class uabPreprocSplit(uabPreprocClass):
    def __init__(self, runChannels, extension, description, chanId, name = 'TileChanSplit'):
        #runChannels is an index into the list of tile-maps that exist for this collection.  This is different from the chanId which is an index into the channels of the particular tile-map being processed in this block.  For example, runChannels = 1 could refer to the RGB image and chanId=2 refers to the Green channel of this tile.
        super(uabPreprocSplit, self).__init__(runChannels, name, extension, description)
        self.channelToSave = chanId
        
    
    def algoName(self):
        return '%s_chan%d' % (self.name, self.channelToSave)
        
    def runTilePreproc(self, colObj):
        
        if(type(self.runChannels) is not list):
            self.runChannels = [self.runChannels]
        
        for tileChanId in self.runChannels:
            """
            ext, _ = colObj.getExtensionInfoById(tileChanId)
            extSpl = ext.split('.')
            extUse = extSpl[0] + str(self.channelToSave) + '.' + extSpl[1]
            """
            for tile in colObj.dataListForRun:
                
                path = os.path.join(self.getDirectoryPaths(colObj), tile + '_' + self.ext)
                code = util_functions.read_or_new_pickle(path)
                
                if(code == 0):
                    im = colObj.loadTileDataByExtension(tile, tileChanId)
                    assert(len(im.shape) == 3)
                    imSplit = np.squeeze(im[:,:,self.channelToSave])
                    util_functions.read_or_new_pickle(path,toSave=1, variable_to_save = imSplit)

#class to perform an operation on two tiles (e.g., difference of two)                    
class uabPreprocMultChanOp(uabPreprocClass):
    def __init__(self, runChannels, extension, description, chans, opDetails, name = 'MultChanOp'):
        #runChannels is an index into the list of tile-maps that exist for this collection.  chans are the indexes of channels to process.  opDetails is a class of type operator (see uabPreProcClasses.py)
        super(uabPreprocMultChanOp, self).__init__(runChannels, name, extension, description)
        self.opChans = chans
        self.opDet = opDetails
    
    def algoName(self):
        return '%s_chans%s_%s' % (self.name, '-'.join([str(a) for a in self.opChans]), self.opDet.getName())
        
    def runTilePreproc(self, colObj):
        #for each tile, apply the operation (e.g., rescaling).  Check whether this exists otherwise, call the tile operator
        for tile in colObj.dataListForRun:
            path = os.path.join(self.getDirectoryPaths(colObj), tile + '_' + self.ext)
            code = util_functions.read_or_new_pickle(path)
            
            if(code == 0):
                #get a list of all the tiles to operate on based on the specified channels to run on 
                tileData = []
                for tileChanId in self.opChans:
                    tileData.append(colObj.loadTileDataByExtension(tile, tileChanId))
                
                opTile = self.opDet.run(tileData)

                util_functions.read_or_new_pickle(path,toSave=1, variable_to_save = opTile)
