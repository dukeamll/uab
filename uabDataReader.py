# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 16:46:12 2017

@author: Daniel

class that handles the reading and iterating over files to be compatible with tensorflow.  This handles constructing a queue or an iterator in a way transparent to the user.  Assumes that a thread & session already exist to load and return data.

call readerAction() to get your data for training/testing [whether it's a queue or reader is handled internally]
"""

import skimage.transform
import scipy.misc, os
import numpy as np
import tensorflow as tf
import uabUtilreader
import util_functions


#class to load all the possible slices of your data    
class ImageLabelReader(object):
    
    def __init__(self, gtInds, dataInds, parentDir, chipFiles, chip_size, tile_size,  batchSize, nChannels = 1,
                 overlap=0, padding=np.array((0,0)), block_mean=None, dataAug='', random=True, isTrain = True):
        # gtInds - indexes to the extensions of the ground truth.  Refers to the indexes from the collection
        # dataInds - indexes to the extensions of the data.  Refers to the indexes from the collection
        # parentDir - directory with the data (training: probably the output of the patch-extractor. testing: probably the raw data directory)
        # chipFiles - list of patch/tile names.  Could also be the file list output from the patch-extractor
        # chip-size - data-size of input to the network
        # tile-size - the data size of the file specified in chipFiles (e.g., tile or chip).  If this is different from chip-size, then a patch extraction operation (with overlap & padding) occurs to operate on the data piecemeal.
        # batchSize - number of chips to send to the network
        # nChannels - number of channels for each input file
        # overlap - in pixels (1 side), single number
        # padding - in pixels (1 side), tuple (y,x)
        # block_mean: mean value of each channel
        # dataAug - augmentation (supports 'flip', 'rotate')
        # random - order in which files are provided to the network
        # isTrain - this pertains only to iterators (not queues).  The iterator needs an infinite loop during training or the resource gets exhausted.
        
        self.chip_size = chip_size
        self.block_mean = block_mean
            
        # chipFiles:
        # list of lists.  Each inner list is a list of the chips by their extension.  These are all the input feature maps for a particular tile location
        # need to separate the file names into their own vectors
        
        if(isinstance(chipFiles, str)):
            filename = os.path.join(parentDir,chipFiles) 
            with open(filename) as file:
                chipFiles = file.readlines()
            
            chipFiles = [a.strip().split(' ') for a in chipFiles]
        self.chip_files = chipFiles
        
        if(nChannels is not list):
            self.nChannels = [nChannels for a in range(len(chipFiles[0]))]
        else:
            self.nChannels = nChannels
                       
        el1 = chipFiles[0]
        if(type(gtInds) is not list):
            gtInds = [gtInds]
            
        self.gtInds = gtInds
        self.dataInds = dataInds
            
        procInds = self.gtInds + self.dataInds
        # reorder the elements in el1 to get the extensions in the right order
        el1 = [el1[i] for i in procInds]
        self.nChannels = [self.nChannels[i] for i in procInds]
        
        # need to decide whether this can be a queue based or a regular data-iterator.
        # Can only use a queue if all the files are jpg/png otherwise need to use the slower data-reader
        self.fileExts = [a.split('.')[-1] for a in el1]
        extExistence = [a in ['jpg','png','jpeg'] for a in self.fileExts]
        if(all(extExistence) and isTrain == 1):
            self.isQueue = 1

            fnameList = []
            for i in procInds:
                fnameList.append(tf.convert_to_tensor([os.path.join(parentDir,a[i]) for a in chipFiles], dtype=tf.string))
                
            self.internalQueue = tf.train.slice_input_producer(fnameList, shuffle=random)
            self.dataLists = self.readFromDiskQueue(dataAug)
            self.readManager = self.dequeue(batchSize)
        else:
            self.isQueue = 0
                
            if(isTrain):
                self.readManager = self.readFromDiskIteratorTrain(parentDir, chipFiles, batchSize, self.chip_size, random, padding, dataAug)
            else:
                self.readManager = self.readFromDiskIteratorTest(parentDir, chipFiles, batchSize, tile_size, self.chip_size, overlap, padding)
    
    def readerAction(self, sess=None):
        if self.isQueue:
            return sess.run(self.readManager)
        else:
            return next(self.readManager)
    
    def readFromDiskIteratorTrain(self, image_dir, chipFiles, batch_size, patch_size, random, padding=(0,0), dataAug=''):
        # this is a iterator for training
        
        if(random):
            idx = np.random.permutation(len(chipFiles))
        else:
            idx = np.arange(stop=len(chipFiles))
        nDims = len(chipFiles[0])
        while True:
            image_batch = np.zeros((batch_size, patch_size[0], patch_size[1], nDims))
            for cnt, randInd in enumerate(idx):
                row = chipFiles[randInd]
                blockList = []
                nDims = 0
                for file in row:
                    img = util_functions.uabUtilAllTypeLoad(os.path.join(image_dir,file))
                    if len(img.shape) == 2:
                        img = np.expand_dims(img, axis=2)
                    nDims += img.shape[2]
                    blockList.append(img)
                block = np.dstack(blockList)

                if self.block_mean is not None:
                    block -= self.block_mean

                if dataAug != '':
                    augDat = uabUtilreader.doDataAug(block, nDims, dataAug, img_mean=self.block_mean, is_np=True)
                else:
                    augDat = block
            
                if (np.array(padding) > 0).any():
                    augDat = uabUtilreader.pad_block(augDat, padding)
                
                image_batch[cnt % batch_size, :, :, :] = augDat
               
                if((cnt+1) % batch_size == 0):
                    yield image_batch[:, :, :, 1:], image_batch[:, :, :, :1]
    
    def readFromDiskIteratorTest(self, image_dir, chipFiles, batch_size, tile_dim, patch_size, overlap=0, padding=(0,0)):
        # this is a iterator for test
        for row in chipFiles:
            blockList = []
            nDims = 0
            for cnt, file in enumerate(row):
                if type(image_dir) is list:
                    img = util_functions.uabUtilAllTypeLoad(os.path.join(image_dir[cnt], file))
                else:
                    img = util_functions.uabUtilAllTypeLoad(os.path.join(image_dir, file))
                if len(img.shape) == 2:
                    img = np.expand_dims(img, axis=2)
                nDims += img.shape[2]
                blockList.append(img)
            block = np.dstack(blockList).astype(np.float32)

            if not np.all([np.array(tile_dim) == block.shape[:2]]):
                block = skimage.transform.resize(block, tile_dim, order=0, preserve_range=True, mode='reflect')

            if self.block_mean is not None:
                block -= self.block_mean
        
            if (np.array(padding) > 0).any():
                block = uabUtilreader.pad_block(block, padding)
                tile_dim = tile_dim + padding*2
            
            ind = 0
            image_batch = np.zeros((batch_size, patch_size[0], patch_size[1], nDims))
            for patch in uabUtilreader.patchify(block, tile_dim, patch_size, overlap=overlap):
                # print(str(ind) +': '+ str(patch.shape))
                image_batch[ind, :, :, :] = patch
                ind += 1
                if ind == batch_size:
                    ind = 0
                    yield image_batch
            # yield the last chunk
            if ind > 0:
                yield image_batch[:ind, :, :, :]
    

    def dequeue(self, num_elements):
        #puts the images in the queue using the iterator
        batches = tf.train.batch(self.dataLists, num_elements)
        return batches[:, :, :, 1:], batches[:, :, :, :1]
    
    def readFromDiskQueue(self, dataAug=''):
        # actually read images from disk
        # apply augmentations (string interface)
        # last input is always the label
        
        queueOutput = []
        totChannels = np.sum(self.nChannels)
        for i in range(len(self.internalQueue)):
            qData = tf.read_file(self.internalQueue[i])
            # depending on whether it is a jpeg or png load it that way
            if(self.fileExts[i] == 'jpg'):
                ldData = tf.image.decode_jpeg(qData, channels=self.nChannels[i])
            elif(self.fileExts[i] == 'png'):
                ldData = tf.image.decode_png(qData, channels=self.nChannels[i])
            else:
                ldData = tf.image.decode_image(qData, channels=self.nChannels[i])
                
            rldData = tf.image.resize_images(ldData, self.chip_size)
            queueOutput.append(rldData)
        queueOutput = tf.squeeze(tf.stack(queueOutput, axis=2), axis=-1)

        if self.block_mean is not None:
            queueOutput -= self.block_mean
            
        if len(dataAug) > 0:
            augDat = uabUtilreader.doDataAug(queueOutput, totChannels, dataAug, img_mean=self.block_mean)
            augDat = tf.image.resize_images(augDat, self.chip_size)
        else:
            augDat = queueOutput
        
        return [augDat]


class ImageLabelReader_cifar(ImageLabelReader):
    def __init__(self, batch_size, patch_size, is_train=True):
        self.isQueue = 0
        self.readManager = self.readFromDiskIteratorTrain(batch_size, patch_size, is_train)

    def readFromDiskIteratorTrain(self, batch_size, patch_size, is_train):
        import keras
        (x_train, _), (x_test, _) = keras.datasets.cifar10.load_data()
        if is_train:
            data = x_train
            n_data = data.shape[0]
        else:
            data = x_test
            n_data = data.shape[0]

        idx = np.random.permutation(n_data)
        while True:
            image_batch = np.zeros((batch_size, patch_size[0], patch_size[1], 3)).astype(np.float32)
            for cnt, randInd in enumerate(idx):
                block = data[randInd, :, :, :]
                image_batch[cnt % batch_size, :, :, :] = scipy.misc.imresize(block, patch_size)
                if ((cnt + 1) % batch_size == 0):
                    yield image_batch, None


# for debugging purposes
if __name__ == '__main__':    
    '''parentDir = '/media/ei-edl01/data/remote_sensing_data/Results/PatchExtr/inria_orgd/chipExtrReg_cSz572x572'
    
    coord = tf.train.Coordinator()
    batch_size = 10
    imR = ImageLabelReader(0, [1, 2], parentDir, 'fileList.txt', (572, 572), (572, 572), batch_size, isTrain=True)

    testBatchSize = 11
    print('initialized reader')
    with tf.Session() as sess:
        
        init = tf.global_variables_initializer()
        sess.run(init)
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        N1 = imR.readerAction(sess)
        #N2 = imR.readerAction(testBatchSize, sess)
        #N3 = imR.readerAction(testBatchSize, sess)
        
        coord.request_stop()
        coord.join(threads)
        
    import matplotlib.pyplot as plt
    for i in range(5):
        plt.subplot(5,3,i*3+1)
        plt.imshow(N1[i,:,:,0])
        plt.subplot(5,3,i*3+2)
        plt.imshow(N1[i,:,:,1])
        plt.subplot(5,3,i*3+3)
        plt.imshow(N1[i,:,:,2])
    plt.show()'''

    im_reader = ImageLabelReader_cifar(6, (128, 128), True)
    data, _ = im_reader.readerAction()
    import matplotlib.pyplot as plt
    for i in range(6):
        plt.subplot(231+i)
        plt.imshow(data[i, :, :, :])
    plt.show()
