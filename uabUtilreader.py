#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 10:18:31 2017

@author: Daniel

Useful functions for data readers.  Particularly, holds the functions for augmentation which should be accessed through doDataAug()
"""

import tensorflow as tf


def block_flipping(block):
    return tf.image.random_flip_left_right(tf.image.random_flip_up_down(block))


def block_rotating(block):
    random_times = tf.to_int32(tf.random_uniform([1], minval=0, maxval=4))[0]
    return tf.image.rot90(block, random_times)


def image_flipping(img, nDim):
    """
    randomly flips images left-right and up-down
    :param img:
    :return:flipped images
    """
    temp = tf.cast(img, dtype=tf.float32)
    temp_flipped = block_flipping(temp)
    img = tf.slice(temp_flipped, [0, 0, 0], [-1, -1, nDim])
    return img


def image_rotating(img, nDim):
    """
    randomly rotate images by 0/90/180/270 degrees
    :param img:
    :param label:
    :return:rotated images
    """
    temp_rotated = block_rotating(img)
    img = tf.slice(temp_rotated, [0, 0, 0], [-1, -1, nDim])
    return img

def doDataAug(data, dataMeta, augType):
    #function to call that actually performs the augmentations
    #dataMeta is a list of info (e.g. label, number of channels)
    
    if 'flip' in augType:
        image = image_flipping(data, dataMeta[1])
    if 'rotate' in augType:
        image = image_rotating(data, dataMeta[1])
    
    return image