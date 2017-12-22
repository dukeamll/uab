"""
Created on 12/16/2017
This file show examples of following steps:
    1. Make a collection on inria with RGB data
    2. Modify the GT and map it to (0, 1)
    3. Extract patches of given size
    4. Make train and validation folds
    4. Train a UNet on those patches
    5. Load a first 7 layers of pretrained model on topcoder data and fine tune for few epochs
"""

import os
import time
import numpy as np
import tensorflow as tf
import uabDataReader
import uabRepoPaths
import uabCrossValMaker
import bohaoCustom.uabPreprocClasses as bPreproc
import uabPreprocClasses
import uab_collectionFunctions
import uab_DataHandlerFunctions
from bohaoCustom import uabMakeNetwork_UNet

# experiment settings
chip_size = (572, 572)
tile_size = (5000, 5000)
batch_size = 5                  # mini-batch size
learn_rate = 1e-4               # learning rate
decay_step = 60                 # learn rate dacay after 60 epochs
decay_rate=0.1                  # learn rate decay to 0.1*before
epochs=25                       # total number of epochs to rum
start_filter_num=32             # the number of filters at the first layer
n_train = 8000                  # number of samples per epoch
n_valid = 1000                  # number of samples every validation step
model_name = 'inria_aug'        # a suffix for model name
GPU = 1                         # which gpu to use
pre_trained_model_dir = '/media/ei-edl01/user/bh163/models/dcc/UnetCrop_um_aug_grid_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32'
layers2keep = '1,2,3,4,5,6,7'   # load first 7 layers, use tensorboard to check details

# make network
# define place holder
X = tf.placeholder(tf.float32, shape=[None, chip_size[0], chip_size[1], 3], name='X')
y = tf.placeholder(tf.int32, shape=[None, chip_size[0], chip_size[1], 1], name='y')
mode = tf.placeholder(tf.bool, name='mode')
model = uabMakeNetwork_UNet.UnetModelCrop({'X':X, 'Y':y},
                                          trainable=mode,
                                          input_size=chip_size,
                                          batch_size=batch_size,
                                          learn_rate=learn_rate,
                                          decay_step=decay_step,
                                          decay_rate=decay_rate,
                                          epochs=epochs,
                                          start_filter_num=start_filter_num)
model.create_graph('X', class_num=2)
# If you only want to load a specific number of layers, you have to do load_weight() here
# don't give pretrained_model_dir and layers2keep when calling model.run(), that will cause problem
model.load_weights(pre_trained_model_dir, layers2keep)

# create collection
# the original file is in /ei-edl01/data/uab_datasets/inria
blCol = uab_collectionFunctions.uabCollection('inria')
opDetObj = bPreproc.uabOperTileDivide(255)          # inria GT has value 0 and 255, we map it back to 0 and 1
# [3] is the channel id of GT
rescObj = uabPreprocClasses.uabPreprocMultChanOp([], 'GT_Divide.tif', 'Map GT to (0, 1)', [3], opDetObj)
rescObj.run(blCol)
img_mean = blCol.getChannelMeans([0, 1, 2])         # get mean of rgb info
print(blCol.readMetadata())                         # now inria collection has 4 channels, the last one is GT with (0, 1)

# extract patches
extrObj = uab_DataHandlerFunctions.uabPatchExtr([0, 1, 2, 4], # extract all 4 channels
                                                cSize=chip_size, # patch size as 572*572
                                                numPixOverlap=int(model.get_overlap()/2),  # overlap as 92
                                                extSave=['jpg', 'jpg', 'jpg', 'png'], # save rgb files as jpg and gt as png
                                                isTrain=True,
                                                gtInd=3,
                                                pad=model.get_overlap()) # pad around the tiles
patchDir = extrObj.run(blCol)

# make data reader
chipFiles = os.path.join(patchDir, 'fileList.txt')
# use uabCrossValMaker to get fileLists for training and validation
idx, file_list = uabCrossValMaker.uabUtilGetFolds(patchDir, 'fileList.txt', 'force_tile')
# use first 5 tiles for validation
file_list_train = uabCrossValMaker.make_file_list_by_key(idx, file_list, [i for i in range(6, 37)])
file_list_valid = uabCrossValMaker.make_file_list_by_key(idx, file_list, [i for i in range(0, 6)])

with tf.name_scope('image_loader'):
    # GT has no mean to subtract, append a 0 for block mean
    dataReader_train = uabDataReader.ImageLabelReader([3], [0, 1, 2], patchDir, file_list_train, chip_size, tile_size,
                                                      batch_size, dataAug='flip,rotate', block_mean=np.append([0], img_mean))
    # no augmentation needed for validation
    dataReader_valid = uabDataReader.ImageLabelReader([3], [0, 1, 2], patchDir, file_list_train, chip_size, tile_size,
                                                      batch_size, dataAug=' ', block_mean=np.append([0], img_mean))

# train
start_time = time.time()

model.train_config('X', 'Y', n_train, n_valid, chip_size, uabRepoPaths.modelPath, loss_type='xent')
# don't give pretarined model dir and layers2keep here
model.run(train_reader=dataReader_train,
          valid_reader=dataReader_valid,
          pretrained_model_dir=None,        # train from scratch, no need to load pre-trained model
          isTrain=True,
          img_mean=img_mean,
          verb_step=100,                    # print a message every 100 step(sample)
          save_epoch=5,                     # save the model every 5 epochs
          gpu=GPU,
          tile_size=tile_size,
          patch_size=chip_size)

duration = time.time() - start_time
print('duration {:.2f} hours'.format(duration/60/60))
