# -*- coding: utf-8 -*-
"""
Created on Mon Nov 06 21:25:52 2017

@author: Daniel

class to extract patches from the data-structure representing each image

computes mean for training images and saves it

for the future:
    (1) separate the paths from this file
    (2) have the by instance information as a class?  maybe?
"""

from __future__ import division
import uabUtilSubm
import os, scipy, glob
import numpy as np

class uabCollection(object):
    
    #the directory names for data & results.  These are sub-directories to the collection name itself for now
    dataPath = 'data'
    resPath = 'Results'
    instanceDir = 'collectionInfo'
    
    def __init__(self, colN, gtExt, dataExts):
        #full path to data directory
        self.colName = colN
        self.imDirectory = os.path.join(uabUtilSubm.parentDir, uabCollection.dataPath)
        #extensions to the data, assumes that the tile name is fixed but that the ground truth & the data have the same tile name.  gtExts -> just the extension for ground truth.  data extensions are their own dictionary to be supplied here
        gtDict = {'GT': gtExt}
        self.extensions = dataExts.copy()
        self.extensions.update(gtDict)
        
        #list of the names of tiles for this collection
        
        #this is split by train and test because that's how the collection was organized but this can be overwritten later
        self.tileTrList = self.getImLists('training')
        self.tileTeList = self.getImLists('testing')
        
        #full list
        self.tileList = self.tileTrList + self.tileTeList
        
        #get the directory where the by-instance information is saved for each tile
        self.instanceMD = os.path.join(self.getResultsDir(), uabCollection.instanceDir)
        kk = dataExts.keys()
        im = self.loadTileData(os.path.join('training', self.tileTrList[0]), kk[0])
        self.tileSize = im.shape
        #self.colMean = self.computeTrainingMean()
        #self.classProps = self.computeClassProportions()
    
    def getImLists(self, dirn):
        #returns the name of all the images in training directory.  Removes the path & extension
        
        kk = self.extensions.values()
        dataSuffs = filter(lambda x: x != self.extensions['GT'], kk)
        
        gtlF = glob.glob(os.path.join(self.imDirectory, dirn , '*' + dataSuffs[0]))
        return [a.split(uabUtilSubm.sl)[-1].split(dataSuffs[0])[0] for a in gtlF]
    
    def getDataNameByTile(self, tileName, ext):
        #convenience function to associate tile name with corresponding data
        return tileName + self.extensions[ext]
    
    def getAllTileByDirAndExt(self, dirn, ext):
        imlist = self.getImLists(dirn)
        return [item + self.extensions[ext] for item in imlist]
    
    def getInstanceInfoByTile(self, tileName, mode, toLoad):
        #this loads in the instance level information for the given tile.  This information was generated using external functions which are not being systematized in this class because they are so dataset dependent.  
        #input: tileName, whether to load or just check existence, mode = ('training', testing')
        #output: toLoad = 0 -> info exists? yes-1, no- 0
        #        toLoad = 1 -> info exists? yes-object, no- []
        
        #The output is for a tile is a datastructure with a list of valid pixels at which to extract patches & their label
        
        #make name
        infName = tileName + '_GTlocs.pkl'
        
        fOutput = os.path.join(self.instanceMD, mode, infName)
        code = uabUtilSubm.read_or_new_pickle(fOutput, toLoad, 0)
        
        return code
    
    def getResultsDir(self):
        return os.path.join(uabUtilSubm.parentDir, uabCollection.resPath)
    
    def getDataExtensions(self):
        #remove the ground truth extension and return all the rest
        return [a for a in self.extensions.keys() if a != 'GT']
    
    def loadTileData(self, tileName, ext):
        imNm = os.path.join(self.imDirectory, self.getDataNameByTile(tileName, ext))
        if imNm[-3:] != 'npy':
            return scipy.misc.imread(imNm)
        else:
            return np.load(imNm)
    
    def loadGTdata(self, tileName):
        return scipy.misc.imread(os.path.join(self.imDirectory, self.getDataNameByTile(tileName, 'GT')))
    
    def computeTrainingMean(self, forceRun=0):
        #computes the mean RGB value of the pixels in training
        
        resDir = os.path.join(self.getResultsDir(), 'collectionInfo','training')
        fname = self.colName+'_mean'
        rgb = uabUtilSubm.read_or_new_pickle(os.path.join(resDir, fname), toLoad=1)
        
        if(not(isinstance(rgb, np.ndarray))):
            #make the directory for the results
            try:
                os.makedirs(resDir)
            except:
                pass
            
            rgb = np.zeros((1,3))
            ind = 0
            for tilename in self.tileTrList:
                cIm = self.loadTileData('training' + uabUtilSubm.sl + tilename)
                cr = cIm.reshape(cIm.shape[0]*cIm.shape[1],cIm.shape[2])
                rgb += np.mean(cr, axis=0)
                ind += 1
                if(np.mod(ind,10) == 0):
                    print 'Finished tile %d' % (ind)
            
            rgb /= len(self.tileTrList)
            
            uabUtilSubm.read_or_new_pickle(os.path.join(resDir, fname), toSave=1,variable_to_save=rgb)
        
        return rgb
    
    def computeClassProportions(self, forceRun=0):
        #computes the class proportions of the pixels in training
        resDir = os.path.join(self.getResultsDir(), 'collectionInfo','training')
        fname = self.colName+'_classProps'
        cprops = uabUtilSubm.read_or_new_pickle(os.path.join(resDir, fname), toLoad=1)
        
        if(not(isinstance(cprops, np.ndarray))):
            #make the directory for the results
            try:
                os.makedirs(resDir)
            except:
                pass
            
            cprops = np.zeros(2)
            ind = 0
            totalPixels = np.prod(self.tileSize)
            for tilename in self.tileTrList:
                cIm = self.loadGTdata('training' + uabUtilSubm.sl + tilename)
                cIm[cIm != 0] = 1
                numTarget = sum(cIm.flatten())
                cprops[0] += (totalPixels - numTarget) / totalPixels
                cprops[1] += numTarget / totalPixels
                
                ind += 1
                if(np.mod(ind,10) == 0):
                    print 'Finished tile %d' % (ind)
            
            cprops /= len(self.tileTrList)
            
            uabUtilSubm.read_or_new_pickle(os.path.join(resDir, fname), toSave=1,variable_to_save=cprops)
            
        return cprops
    
    
class uabCollection_bh(uabCollection):
    
    #the directory names for data & results.  These are sub-directories to the collection name itself for now
    dataPath = 'data'
    resPath = 'Results'
    instanceDir = 'collectionInfo'
    
    def __init__(self, colN, gtExt, dataExts):
        #full path to data directory
        self.colName = colN
        self.imDirectory = os.path.join(uabUtilSubm.parentDir, uabCollection.dataPath)
        #extensions to the data, assumes that the tile name is fixed but that the ground truth & the data have the same tile name.  gtExts -> just the extension for ground truth.  data extensions are their own dictionary to be supplied here
        gtDict = {'GT': gtExt}
        self.extensions = dataExts.copy()
        self.extensions.update(gtDict)
        
        #list of the names of tiles for this collection
        
        #this is split by train and test because that's how the collection was organized but this can be overwritten later
        self.tileTrList = self.getImLists('training')
        self.tileTeList = self.getImLists('testing')
        
        #full list
        self.tileList = self.tileTrList + self.tileTeList
        
        #get the directory where the by-instance information is saved for each tile
        self.instanceMD = os.path.join(self.getResultsDir(), uabCollection.instanceDir)
        self.tileSize = np.array([2048, 2048])
        #kk = dataExts.keys()
        #im = self.loadTileData(os.path.join('testingPadded', self.tileTrList[0]), kk[0])
        #self.tileSize = im.shape
        #self.colMean = self.computeTrainingMean()
        #self.classProps = self.computeClassProportions()
    
        
            
        