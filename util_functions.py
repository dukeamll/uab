# -*- coding: utf-8 -*-
"""
Created on Mon Nov 06 17:55:23 2017

@author: Daniel

some globally utile functions
"""

from __future__ import division
import scipy
from sys import platform
import os
import numpy as np

sl = os.path.sep

if platform == 'win32':
    #this is the top-level directory with data & results obviously should be changed
    parentDir = 'Y:\\data\\'
elif platform == 'linux2':
    parentDir = '/media/Y/data/'    
    
def uabUtilMakeDirectoryName(dirName):
    if(not(os.path.isdir(dirName))):
        os.makedirs(dirName)     

def read_or_new_pickle(path, toLoad = 0, toSave = 0, variable_to_save = []):
    #check whether a pickled file exists
    #load if it exists otherwise return -1
    #
    #can't have toLoad & toSave set simultaneously
    #if toLoad=0 and toSave=0 then just checks if file exists 
    #exists? 1: 0
    #
    #if toLoad = 1
    #if file exists, return object else -1
    #
    #if toSave = 1
    #whether file exists or not, overwrite. return 2
    
    
    if(toLoad == 1 and toSave == 1):
        #race condition
        raise NameError('Cannot run function this way')
    
    if path and os.path.isfile(path):    
        #if the file exists and you wanted to load it
        if(toLoad == 1):
            return uabUtilAllTypeLoad(path)
        else:
            code = 1
    else:
        code = 0
    
    if(toSave == 1):
        uabUtilAllTypeSave(path, variable_to_save)
        code = 2
        
    return code

def uabUtilAllTypeLoad(fileName):
    #handles the loading of a file of all types in python
    
    try:
        if fileName[-3:] != 'npy':
            outP = scipy.misc.imread(fileName)
        else:
            outP = np.load(fileName)
        
        return outP
    except StandardError: # so many things could go wrong, can't be more specific.
        raise NotImplementedError('Problem loading this data tile')

def uabUtilAllTypeSave(fileName, variable_to_save):
    #handles the loading of a file of all types in python
    if fileName[-3:] != 'npy':
        scipy.misc.imsave(fileName, variable_to_save)
    else:
        np.save(fileName, variable_to_save) 

def d2s(decimal, ndigs=5):
    #input decimal, returns it as a string with the dot replaced by a 'p'
    inpStr = '%0.'+('%d'%ndigs)+'f'
    replStr = inpStr % decimal
    return replStr.replace('.','p')

def data_iterator(batch_size, patchList, toRotate=1, toFlip=1):
    """
    Randomly load and iterate small patches
    :param patchList: list of patches to use for training/validation
    :param batch_size: number of patch to load each time
    :return: images and labels batch
    """
    
    nPatches = len(patchList)
    idx = np.random.permutation(nPatches)
    while True:
        # random pick num of batch_size images
        for i in range(0, nPatches, batch_size):
            images_batch, labels_batch = read_data([patchList[j] for j in idx[i:i+batch_size]], toRotate, toFlip)
            yield images_batch, labels_batch    

def read_data(data_list, random_rotation=False, random_flip=False):
    """
    read stored patch file and augment data
    :param data_list: list of patch files to read
    :param random_rotation: boolean, True for random rotation
    :param random_flip: boolean, True for random filp
    :return: stack of data n*w*h*c
    """
    patch_num = len(data_list)

    # get width and height
    test_data = scipy.misc.imread(data_list[0][0])
    w, h, c = test_data.shape

    data_chunk = np.zeros((patch_num, w, h, c))
    labels_chunk = np.zeros((patch_num, w, h, 1))
    for i in range(patch_num):
        data_chunk[i,:,:,:] = data_augmentation(scipy.misc.imread(data_list[i][0]),                                               random_rotation=random_rotation, random_flip=random_flip)
        labels_chunk[i,:,:,:] = data_augmentation(scipy.misc.imread(data_list[i][1]),                                               random_rotation=random_rotation, random_flip=random_flip)
        
    return data_chunk, labels_chunk  

def data_augmentation(data, random_rotation=False, random_flip=False):
    """
    Augment data by random rotate or flip
    :param data: stack of 3D image with a dimension of n*w*h*c (c can be greater than 3)
    :param random_rotation: boolean, True for random rotation
    :param random_flip: boolean, True for random flip
    :return:
    """
    if random_rotation:
        direction = np.array([-1, 1])
        direction = direction[np.random.randint(2)]
        _, _, c = data.shape
        for i in range(c):
            data[:,:,i] = np.rot90(data[:,:,i], direction)
    if random_flip:
        direction = np.random.randint(2)
        if direction == 1:
            data = data[:,::-1,:]
        else:
            data = data[::-1,:,:]
    return data          

def get_pred_labels(pred):
    """
    Get predicted labels from the output of CNN softmax function
    :param pred: output of CNN softmax function
    :return: predicted labels
    """
    n, h, w, c = pred.shape
    outputs = np.zeros((n, h, w, 1), dtype=np.uint8)
    for i in range(n):
        outputs[i] = np.expand_dims(np.argmax(pred[i,:,:,:], axis=2), axis=2)
    return outputs


def decode_labels(label, num_images=10):
    """
    Decode label back to RGB data
    :param label: image with elements corresponding to category number
    :param num_images: num of images to decode
    :return: image with elements corresponding to RGB data
    """
    n, h, w, c = label.shape
    outputs = np.zeros((n, h, w, 3), dtype=np.uint8)
    label_colors = [(255,255,255),(0,0,255)]
    for i in range(num_images):
        pixels = np.zeros((h, w, 3), dtype=np.uint8)
        for j in range(h):
            for k in range(w):
                pixels[j,k] = label_colors[np.int(label[i,j,k,0])]
        outputs[i] = pixels
    return outputs