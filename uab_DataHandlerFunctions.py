# -*- coding: utf-8 -*-
"""
Created on Wed Nov 08 17:03:03 2017

@author: Daniel

Class to handle the extraction of chips from the tiles
has a name & a predictable place to store output.
 
Note: if you want to do some custom augmentation, this would also be the place to do it [save multiple chips at each location].  Don't forget to update the save name accordingly

Default
[a] Extract every possible tile at regular intervals with minimal overlap.  The amount of overlap is an input to the patch extractor class
[b] saves extracted patches using the same file-extension (e.g., tif) as that in the collection files.  If you want this to change, then input a list of extensions into the constructor in the same order as the channels defined in runChannels.

[Note: The default is an extractor that does not sample the tile densely.  This is because the FCNs already see many shifted copies of the data, so seeing more is not very beneficial.]

If you want to extract *fewer* patch, then make a child-class that overloads makeGrid() where a list of coordinates at which to extract patches is defined

Instructions for overloading
(1) algoName()
(2) makeGrid()
(3) extrName()

"""

import util_functions, os
import numpy as np
import tensorflow as tf
import uabBlockparent
from uabBlockparent import uabBlock

class uabPatchExtr(uabBlock):
    
    fname = 'fileList.txt'
    verbStep = 10
    
    def __init__(self, runChannels, name='Reg', cSize=(224,224), numPixOverlap = 0,extSave=None):
        super(uabPatchExtr, self).__init__(runChannels, name)
        #chip size
        self.chipExtrSize = cSize
        #numPixOverlap -> number of pixels to overlap the patches by.  If = 0, then extract tiles such that the first patch starts on row 1 and the final patch ends on the final row (& dito for columns)
        self.numPixOverlap = numPixOverlap
        self.coord = tf.train.Coordinator()
        self.saveExts = extSave
    
    def extrName(self):
        return ''
    
    def algoName(self):
        #name starter for all patch extractors
        strName = self.extrName()
        if(strName):
            strName = '_' + strName
        else:
            strName = ''
                
        return 'chipExtr%s_cSz%dx%d%s' % (self.name, self.chipExtrSize[0], self.chipExtrSize[1], strName)
    
    def getDirectoryPaths(self, colObj):
        return uabBlock.getBlockDir(os.path.join(uabBlockparent.outputDirs['patchExt'], colObj.colName, self.algoName()))
    
    def makeGrid(self, tileSz):
        #this function should be changed in the subclass if desired.  Default behavior is to extract chips at fixed locations.  Output coordinates for Y,X as a list (not two lists)
        
        #make the grid of indexes at which to extract patches
        #get the boundary of the tile given the patchsize
        maxIm0 = tileSz[0] - self.chipExtrSize[0] - 1 
        maxIm1 = tileSz[1] - self.chipExtrSize[1] - 1
        if(self.numPixOverlap == 0):
            #this is to extract with as little overlap as possible
            DS0 = np.ceil(tileSz[0]/self.chipExtrSize[0])
            DS1 = np.ceil(tileSz[1]/self.chipExtrSize[1])
            patchGridY = np.floor(np.linspace(0,maxIm0,DS0))
            patchGridX = np.floor(np.linspace(0,maxIm1,DS1))
        elif(self.numPixOverlap > 0):
            #overlap by this number of pixels
            #add the last possible patch to ensure that you are covering all the pixels in the image
            patchGridY = list(range(0, maxIm0, self.chipExtrSize[0] - self.numPixOverlap))
            patchGridY = patchGridY + [maxIm0]
            patchGridX = list(range(0, maxIm1, self.chipExtrSize[1] - self.numPixOverlap))
            patchGridX = patchGridX + [maxIm1]
        
        Y,X = np.meshgrid(patchGridY,patchGridX)
        return list(zip(Y.flatten(),X.flatten()))
    
    
    def runAction(self, colObj):
        #function to extract the chips from the tiles
        
        gridList = self.makeGrid(colObj.tileSize)
        
        directory = self.getDirectoryPaths(colObj)
        #extract chips for all the specified extensions
        
        #precompute extensions
        fileExts = []
        for cnt, chanId in enumerate(self.runChannels):
            ext,_ = colObj.getExtensionInfoById(chanId)
            if(self.saveExts is not None):
                sExt = ext.split('.')
                fileExts.append(sExt[0] + '.' + self.saveExts[cnt])
            else:
                fileExts.append(ext)
        
        with open(os.path.join(directory,uabPatchExtr.fname),'w') as file:
            for ind, tilename in enumerate(colObj.dataListForRun):
                for coordList in gridList:
                    #extract patches for all the channels at coordinate location.  This is done so that the file containing patch names can have all the extracted patches of one location on a single line
                    nmStr = []
                    
                    for ext, chanId in zip(fileExts, self.runChannels): 
                        #get the output file's name
                        x1 = int(coordList[0])
                        x2 = int(coordList[1])
                        finNm = tilename+'_y%dx%d_%s'%(x1, x2,ext)
                        nmStr.append(finNm)
                        
                        fPath = os.path.join(directory, finNm)
                        isExt = util_functions.read_or_new_pickle(fPath, toLoad=0)
                        if(isExt == 0):
                            cIm = colObj.loadTileDataByExtension(tilename, chanId)
                            
                            #extract a patch from the image
                            nDims = cIm.shape
                            if(len(nDims) == 2):
                                chipDat = cIm[x1:x1+self.chipExtrSize[0], x2:x2 + self.chipExtrSize[1]]
                            else:
                                chipDat = cIm[x1:x1+self.chipExtrSize[0], x2:x2 + self.chipExtrSize[1],:]
                            
                            util_functions.read_or_new_pickle(fPath, toSave = 1, variable_to_save=chipDat)
                        
                    file.write("%s\n" % ' '.join(nmStr))
                            
                if(np.mod(ind, uabPatchExtr.verbStep) == 0):
                    print('Finished tile %d' % (ind))