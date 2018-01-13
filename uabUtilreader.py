#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 10:18:31 2017

@author: Daniel

Useful functions for data readers.  Particularly, holds the functions for augmentation which should be accessed through doDataAug()
"""


import tensorflow as tf
import numpy as np

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


def image_rotating_np(img):
    """
    randomly rotate images by 0/90/180/270 degrees
    :param img:
    :param label:
    :return:rotated images
    """
    rot_time = np.random.randint(low=0, high=4)
    img = np.rot90(img, rot_time, (0, 1))
    return img


def image_flipping_np(img):
    """
    randomly flips images left-right and up-down
    :param img:
    :return:flipped images
    """
    v_flip = np.random.randint(0, 1)
    h_flip = np.random.randint(0, 1)
    if v_flip == 1:
        img = img[::-1, :, :]
    if h_flip == 1:
        img = img[:, ::-1, :]
    return img


def doDataAug(data, dataMeta, augType, is_np=False):
    #function to call that actually performs the augmentations
    #dataMeta is a list of info (e.g. label, number of channels)

    if is_np:
        if 'flip' in augType:
            data = image_flipping_np(data)
        if 'rotate' in augType:
            data = image_rotating_np(data)
    else:
        if 'flip' in augType:
            data = image_flipping(data, dataMeta)
        if 'rotate' in augType:
            data = image_rotating(data, dataMeta)
    
    return data

def crop_image(block, size, corner):
    return block[corner[0]:corner[0]+size[0],corner[1]:corner[1]+size[1],:]

def patchify(block, tile_dim, patch_size, overlap=0):
    max_h = tile_dim[0] - patch_size[0]
    max_w = tile_dim[1] - patch_size[1]
    if max_h > 0 and max_w > 0:
        h_step = np.ceil(tile_dim[0] / (patch_size[0] - overlap))
        w_step = np.ceil(tile_dim[1] / (patch_size[1] - overlap))
    else:
        h_step = 1
        w_step = 1
    patch_grid_h = np.floor(np.linspace(0, max_h, h_step)).astype(np.int32)
    patch_grid_w = np.floor(np.linspace(0, max_w, w_step)).astype(np.int32)
    for corner_h in patch_grid_h:
        for corner_w in patch_grid_w:
            yield crop_image(block, patch_size, (corner_h, corner_w))

def pad_block(block, pad):
    padded_block = []
    _, _, c = block.shape
    for i in range(c):
        padded_block.append(np.pad(block[:, :, i],
                                   ((pad[0].astype(np.int), pad[1].astype(np.int)),
                                   (pad[0].astype(np.int), pad[1].astype(np.int))),
                                   'symmetric'))
    return np.dstack(padded_block)            

def un_patchify(blocks, tile_dim, patch_size, overlap=0):
    _, _, _, c = blocks.shape
    image = np.zeros((tile_dim[0], tile_dim[1], c))
    max_h = tile_dim[0] - patch_size[0]
    max_w = tile_dim[1] - patch_size[1]
    h_step = np.ceil(tile_dim[0] / (patch_size[0] - overlap))
    w_step = np.ceil(tile_dim[1] / (patch_size[1] - overlap))
    patch_grid_h = np.floor(np.linspace(0, max_h, h_step)).astype(np.int32)
    patch_grid_w = np.floor(np.linspace(0, max_w, w_step)).astype(np.int32)

    cnt = 0
    for corner_h in patch_grid_h:
        for corner_w in patch_grid_w:
            cnt += 1
            image[corner_h:corner_h+patch_size[0], corner_w:corner_w+patch_size[1], :] += blocks[cnt-1, :, :, :]
    return image


def un_patchify_shrink(blocks, tile_dim, tile_dim_output, patch_size, patch_size_output, overlap=0):
    _, _, _, c = blocks.shape
    image = np.zeros((tile_dim_output[0], tile_dim_output[1], c))
    max_h = tile_dim[0] - patch_size[0]
    max_w = tile_dim[1] - patch_size[1]
    h_step = np.ceil(tile_dim[0] / (patch_size[0] - overlap))
    w_step = np.ceil(tile_dim[1] / (patch_size[1] - overlap))
    patch_grid_h = np.floor(np.linspace(0, max_h, h_step)).astype(np.int32)
    patch_grid_w = np.floor(np.linspace(0, max_w, w_step)).astype(np.int32)

    cnt = 0
    for corner_h in patch_grid_h:
        for corner_w in patch_grid_w:
            cnt += 1
            image[corner_h:corner_h + patch_size_output[0], corner_w:corner_w + patch_size_output[1], :] += blocks[cnt - 1, :, :, :]
    return image
