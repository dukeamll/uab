# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 16:46:12 2017

@author: Daniel

class that handles the reading and iterating over files to be compatible with tensorflow
"""

import scipy.misc
import numpy as np
import tensorflow as tf
from sisRepo.dataReader import patch_extractor
import uabUtilreader

    
#class to load all the possible slices of your data    
class ImageLabelReader(object):
    
    @staticmethod
    def getTestIterator(image_dir, batch_size, tile_dim, patch_size, overlap, padding=0):
        # this is a iterator for test
        block = scipy.misc.imread(image_dir)
        if padding > 0:
            block = patch_extractor.pad_block(block, padding)
            tile_dim = (tile_dim[0]+padding*2, tile_dim[1]+padding*2)
        cnt = 0
        image_batch = np.zeros((batch_size, patch_size[0], patch_size[1], 3))
        for patch in patch_extractor.patchify(block, tile_dim, patch_size, overlap=overlap):
            cnt += 1
            image_batch[cnt-1, :, :, :] = patch
            if cnt == batch_size:
                cnt = 0
                yield image_batch
        # yield the last chunck
        if cnt > 0:
            yield image_batch[:cnt, :, :, :]
    
    def __init__(self, chipFiles, input_size, coord, chanInf, random=True, dataAug=''):
        #chanInf-> meta data about that input list (e.g., jpeg- 3 channels)
        #list of lists: [['jpg',3],['png',1]]
        #can add an optional divisor as a third input
        
        self.input_size = input_size
        self.coord = coord
        self.listMeta = chanInf
        
        #need to separate the file names into their own vectors
        el1 = chipFiles[0]
        fnameList = []
        for i in range(len(el1)):
            fnameList.append(tf.convert_to_tensor([a[i] for a in chipFiles], dtype=tf.string))
        
        self.queue = tf.train.slice_input_producer(fnameList, shuffle=random)
        self.dataLists = self.readFromDisk(dataAug)

    def dequeue(self, num_elements):
        #puts the images in the queue using the iterator
        batches = tf.train.batch(self.dataLists, num_elements)
        return batches    
    
    def readFromDisk(self, dataAug=''):
        #actually read images from disk
        #apply augmentations (string interface)
        #last input is always the label
        
        queueOutput = []
        for i in range(len(self.queue)):
            qData = tf.read_file(self.queue[i])
            #depending on whether it is a jpeg or png load it that way
            if(self.listMeta[i][0] == 'jpg'):
                ldData = tf.image.decode_jpeg(qData, channels=self.listMeta[i][1])
            elif(self.listMeta[i][0] == 'png'):
                ldData = tf.image.decode_png(qData, channels=self.listMeta[i][1])
                if(len(self.listMeta[i]) == 3):
                    ldData /= self.listMeta[i][2]
            
            rldData = tf.image.resize_images(ldData, self.input_size)
            
            if dataAug:
                augDat = uabUtilreader.doDataAug(rldData, dataAug)
            else:
                augDat = rldData
            
            queueOutput.append(augDat)
        
        return queueOutput

#reader for the UM competition where we also load in custom maps (like if you want to do an operation on the maps.  This is not recommended and should be done in the pre-processing stages)
class ImageReaderHeightOps(ImageLabelReader):
    def __init__(self, chipFiles, input_size, coord, chanInf, random=True, dataAug='', heightMode=''):
        self.heightMode = heightMode
        super(ImageReaderHeightOps, self).__init__(chipFiles, input_size, coord, chanInf, random, dataAug=dataAug)
    
    def dequeue(self, num_elements):
        batches = \
            tf.train.batch(self.dataLists, num_elements)
        if self.height_mode == 'all':
            return [tf.concat([batches[0], batches[1], batches[2]], axis=3), batches[3]]
        elif self.height_mode == 'subtract':
            #RGB, DSM - DTM
            return [tf.concat([batches[0], batches[1]-batches[2]], axis=3), batches[3]]
        elif self.height_mode == 'subtract_all':
            return [tf.concat([batches[0], batches[1], batches[2],
                              batches[1] - batches[2]], axis=3), batches[3]]
    
    def getTestIterator(self, image_dir, batch_size, tile_dim, patch_size, overlap, padding=0):
        # this is a iterator for test
        block = []
        for file in image_dir:
            img = scipy.misc.imread(file)
            if len(img.shape) == 2:
                img = np.expand_dims(img, axis=2)
            block.append(img)
        #block = np.dstack(block)
        if self.heightMode == 'all':
            block = np.dstack(block)
            image_batch = np.zeros((batch_size, patch_size[0], patch_size[1], 5))
        elif self.heightMode == 'subtract':
            block = np.dstack([block[0], block[1]-block[2]])
            image_batch = np.zeros((batch_size, patch_size[0], patch_size[1], 4))
        else:
            block = np.dstack([block[0], block[1], block[2], block[1]-block[2]])
            image_batch = np.zeros((batch_size, patch_size[0], patch_size[1], 6))
    
        if padding > 0:
            block = patch_extractor.pad_block(block, padding)
            tile_dim = (tile_dim[0]+padding*2, tile_dim[1]+padding*2)
        cnt = 0
        #image_batch = np.zeros((batch_size, patch_size[0], patch_size[1], 3))
        for patch in patch_extractor.patchify(block, tile_dim, patch_size, overlap=overlap):
            cnt += 1
            image_batch[cnt-1, :, :, :] = patch
            if cnt == batch_size:
                cnt = 0
                yield image_batch
        # yield the last chunck
        if cnt > 0:
            yield image_batch[:cnt, :, :, :]
    