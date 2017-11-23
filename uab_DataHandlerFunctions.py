# -*- coding: utf-8 -*-
"""
Created on Wed Nov 08 17:03:03 2017

@author: Daniel

Class to handle the extraction of chips (i.e., large patches for an FCN) from the tiles
has a name & a predictable place to store stuff

Default
[a] Extract every possible at regular intervals with minimal overlap.  The amount of overlap can be made variable
[Note: The default is an extractor that does not sample the tile densely.  This is because the FCNs already see many shifted copies of the data, so seeing more is not very beneficial.]
[b] If you have more than just data files, then add to the suffix list

If you want to extract *less* things by default, then make a child-class that overloads extractChips

Instructions for overloading
(1) algoName -> your specific algorithm's name
(2) extractAction -> defines how you want to subselect chips from the tiles 
[see example of a child-class below that extracts patches randomly but ensuring that enough pixels are h1]
"""
from __future__ import division
import util_functions, os, glob, scipy, re
import numpy as np
from uabUtilSubm import sl
import tensorflow as tf
import uabDataReader

class uabDataHandler(object):
    
    resPath = 'patchDataset'
    fname = 'fileList.txt'
    defSuff = {'data': '_dat.jpg', 'GT': '_GT.png'}
    
    def __init__(self, cSize=(224,224), custName = '', addlSuff = {}):
        #size of chips to extract
        self.chipExtrSize = cSize
        self.custName = custName
        self.coord = tf.train.Coordinator()
        
        #if there are more extensions to consider      
        #the names of these extensions should match those in the collection
        if addlSuff:
            self.suff = uabDataHandler.defSuff.copy()
            self.suff.update(addlSuff)
        else:
            self.suff = uabDataHandler.defSuff

    
    def algoName(self):
        #block specific name
        return ''
    
    def setName(self):
        #name starter for all patch extractors
        strName = self.algoName()
        if(strName):
            strName = '_' + strName
        else:
            strName = ''
        
        if(self.custName):
            cName = '_' + self.custName
        else:
            cName = self.custName
                
        return 'chipExtr_cSz%dx%d%s%s' % (self.chipExtrSize[0], self.chipExtrSize[1], strName, cName)
    
    def getBlockDir(self):
        return os.path.join(uabDataHandler.resPath, self.setName())
    
    def getChipsList(self, colObj, isTraining, subselectList={}):
        #get the list of chips and their associated ground truth
        #subselectList -> dictionary to subselect files by city & tile
        #
        #for now only select tiles whose index is below a certain number
        directory = os.path.join(colObj.getResultsDir(), self.getBlockDir())
        if(isTraining):
            extrDir = 'training'
        else:
            extrDir = 'testing'
        
        if subselectList:
            
            cities = subselectList['city'].split(',')
            tiles = [int(a) for a in subselectList['tileInds'].split(',')]
            
            filename = os.path.join(directory, extrDir, uabDataHandler.fname)
            with open(filename) as f:
                mylist = f.read().splitlines() 
            
            stats = [[a, a.split(sl)[-1].split('_')[:3]] for a in mylist]
            datP = [os.path.join(colObj.getResultsDir(), a[0]) for a in stats if a[1][0] in cities and int(a[1][2]) in tiles]
            
            """
            datP = []
            for cCity in subselectList['city'].split(','):
                cityFiles = glob.glob(os.path.join(directory,extrDir,cCity + '*'+self.suff['dat']))
                if(subselectList['gt'] == 1):
                    datP += [a for a in cityFiles if int(a.split(sl)[-1].split('_')[2]) > subselectList['tileInd']]
                else:
                    datP += [a for a in cityFiles if int(a.split(sl)[-1].split('_')[2]) <= subselectList['tileInd']]
              """      
                #bInd = subselectList['tileInd'].split(',')
                #indvals = ['%03d' % int(a) for a in bInd]
                #indStr = '{' + ','.join(indvals) + '}'
                #datP + glob.glob(os.path.join(directory,extrDir,                                                       '_'.join((cCity,'Tile')) + indStr + '*'+self.suff['dat']))
        else:
            datP = glob.glob(os.path.join(directory,extrDir,'*'+self.suff['GT']))
        
        
        cc = [a.split(self.suff['GT'])[0] for a in datP if a.find(self.suff['GT']) != -1]
        
        #make the inputs ready for the image reader
        #(1) file list
        suffKeys = self.suff.values()
        #ensure that the GT suffix is the last one
        dataSuffs = filter(lambda x: x != self.suff['GT'], suffKeys)
        dataSuffs.append(self.suff['GT'])
        fileList = [[bNm + suffix for suffix in dataSuffs] for bNm in cc]
        
        #(2) meta-data array
        fFile = fileList[0]
        metaArr = []
        for ind in range(len(fFile)):
            im = scipy.misc.imread(fFile[ind])
            mm = [fFile[ind].split('.')[-1]]
            dims = im.shape
            if(len(dims) == 2):
                mm.append(1)
            else:
                mm.append(dims[2])
                
            if(ind == len(fFile)-1):
                mm.append(255)
            else:
                mm.append(1)
                
            
            metaArr.append(mm)
        
        return fileList, metaArr

        #return [[bNm+self.suff['dat'], bNm+self.suff['GT']] for bNm in cc]
    
    def makeDataReader(self, colObj, isTraining, batch_size, subselectList={}):
        #this is a necessary object to feed data into the network.  Can be used for training & validation
        #get the list of chips to feed into the network
        outList, metaArr = self.getChipsList(colObj, isTraining, subselectList)
        
        with tf.name_scope('image_loader'):
            reader = uabDataReader.ImageLabelReader(outList, self.chipExtrSize, self.coord, metaArr)
            batch_op_list = reader.dequeue(batch_size)
        
        return reader, batch_op_list
                
    
    def extractChips(self, colObj, isTrain, numPixOverlap = 0, forceRun = 0):
        #function to extract the chips from the tiles
        #if the directory doesn't exist or forceRun=1 then this runs
        #isTrain -> to have a different action for training and testing
        #   Assumes that the collection has a training list defined
        #numPixOverlap -> number of pixels to overlap the patches by.  If = 0, then extract tiles such that the first patch starts on row 1 and the final patch ends on the final row (& dito for columns)
        #
        #Output: the name of the directory for this extraction
        
        
        #from the grid, chips can be extracted.  Depending on whether this is in train/(test, validation) mode it will either save all the chips or only a subsampling
        if(isTrain > 0):
            #could be validation
            tileList = colObj.tileTrList
            pref = 'training' + sl
        elif(isTrain == 0):
            tileList = colObj.tileTeList
            pref = 'testing' + sl
        
        directory = os.path.join(colObj.getResultsDir(), self.getBlockDir(), pref)
        if not os.path.exists(directory) or forceRun == 1:
            #for each tile in the collection, extract chips as per the specifications of the constructor
            
            #make the directory for the results
            try:
                os.makedirs(directory)
            except:
                pass
            
            #make the grid of indexes at which to extract patches
            #get the boundary of the tile given the patchsize
            maxIm0 = colObj.tileSize[0] - self.chipExtrSize[0] - 1 
            maxIm1 = colObj.tileSize[1] - self.chipExtrSize[1] - 1
            if(numPixOverlap == 0):
                #this is to extract with as little overlap as possible
                DS0 = np.ceil(colObj.tileSize[0]/self.chipExtrSize[0])
                DS1 = np.ceil(colObj.tileSize[1]/self.chipExtrSize[1])
                patchGridY = np.floor(np.linspace(0,maxIm0,DS0))
                patchGridX = np.floor(np.linspace(0,maxIm1,DS1))
            elif(numPixOverlap > 0):
                #overlap by this number of pixels
                #add the last possible patch to ensure that you are covering all the pixels in the image
                patchGridY = range(0, colObj.tileSize[0], self.chipExtrSize[0] - numPixOverlap)
                patchGridY = patchGridY + [maxIm0]
                patchGridX = range(0, colObj.tileSize[1], self.chipExtrSize[1] - numPixOverlap)
                patchGridX = patchGridX + [maxIm1]
                
            
            
            ind = 0
            colObj.imDirectory += sl + pref
            for tilename in tileList:
                self.extractAction(colObj, directory, tilename, patchGridY, patchGridX, isTrain)
                ind += 1
                if(np.mod(ind,10) == 0):
                    print 'Finished tile %d' % (ind)
        
            #make a file that has all the file names from this directory
            #don't append the result directory name as that should be platform agnostic
            f = open(os.path.join(directory,uabDataHandler.fname),'w')
            thelist = os.listdir(directory)
            for item in thelist:
                f.write("%s\n" % os.path.join(self.getBlockDir(), pref, item))
            f.close()
        
        return directory
    
    def extractAction(self, colObj, directory, tilename, patchGridY, patchGridX, isTrain):
        for yind in patchGridY:
            for xind in patchGridX:
                nm = os.path.join(directory, tilename+'_y%dx%d'%(yind, xind))
                for kk in colObj.extensions.keys():
                    if(not(kk == 'GT' and isTrain == 0)):
                        #can't do ground truth if it's not training
                        finNm =  nm + self.suff[kk]
                        isExt = util_functions.read_or_new_pickle(finNm, toLoad=1)
                        if(isExt == 0):
                            cImage = colObj.loadTileData(tilename, kk)
                            if(kk == 'GT'):
                                #normalizing, just to be sure
                                cImage[cImage != 0] = 1
                            
                            util_functions.extractPatch(cImage, yind, xind, self.chipExtrSize[0], self.chipExtrSize[1], finNm)
                #if(isTrain > 0):
                #    util_functions.extractPatch(cGT, yind, xind, self.chipExtrSize[0], self.chipExtrSize[1], 1, nm + self.suff['GT'])
    
    
#example subclass for patch extraction    
#extract patches randomly from the image        
class uabDataRandomExtract(uabDataHandler):
    def __init__(self, cSize=(224,224), h1per=0.3,  chipPerTile=20):
        #inherit from superclass
        super(uabDataRandomExtract, self).__init__(cSize)
        #number of chips to ext%autoreloadract per tile.  This defines the amount of overlap required in making the chip grid
        self.nChipsPerTile = chipPerTile
         #what should the final percentage of pixels be in the set of chips that corresponds to targets
        self.percentTarget = h1per
    
    def setName(self):
        return 'nChip%d_h1per%s' % (self.nChipsPerTile, util_functions.d2s(self.percentTarget,1))
    
    def extractAction(self, directory, tilename, cImage, patchGridY, patchGridX, isTrain, cGT = []):
         #select randomly the set of indexes to sample.  The number corresponds to the number of chips 
         #need to check that the proportion of target pixels in the extracted images are above the percentage of pixels defined in the constructor.  Because this is not always possible, every 10 iterations, reduce the number by 1 percent point
        numIter = 0
        while 1 < 2:
            rY     = np.random.permutation(patchGridY)
            rX     = np.random.permutation(patchGridX)
            p1, p2 = np.meshgrid(rY, rX)
            randY  = p1.flatten().reshape(p1.size,1)
            randX  = p2.flatten().reshape(p2.size,1)
            
            gtIm = np.zeros(self.chipExtrSize)
            for ind in range(self.nChipsPerTile):
                #get the sum of all the pixels that correspond to target
                gtIm += util_functions.extractPatch(cGT, randY[ind], randX[ind], self.chipExtrSize[0], self.chipExtrSize[1], 1)
            
            #check that the patches extracted have enough h1 pixels
            rat = sum(gtIm.flatten()) / (self.nChipsPerTile * np.prod(self.chipExtrSize))                       
            print "ratio: %0.4f" % (rat)
            if(rat > self.percentTarget - np.floor(numIter/10)/100):
                for ind in range(self.nChipsPerTile):                            
                    #extract data & ground truth
                    nm = os.path.join(directory, tilename+'_y%dx%d'%(int(randY[ind]), int(randX[ind])))
                    util_functions.extractPatch(cImage, randY[ind], randX[ind], self.chipExtrSize[0], self.chipExtrSize[1], 3, nm + self.suff['dat'])
                    util_functions.extractPatch(cGT, randY[ind], randX[ind], self.chipExtrSize[0], self.chipExtrSize[1], 1, nm + self.suff['GT'])
                break
            else:
                numIter += 1
            
            #if not, then try again