"""
Created on 02/07/2018
Bohao Huang
This file show examples of following steps:
    1. Make a collection on inria with RGB data
    2. Modify the GT and map it to (0, 1)
    3. Extract patches of given size
    4. Load a pretrained model
    5. Test on Inria and save prediction maps
"""
import tensorflow as tf
import uabCrossValMaker
import uab_collectionFunctions
from bohaoCustom import uabMakeNetwork_UNet

# settings
gpu = 0                         # which gpu to use, remember to set to None if you don't know which one to use
batch_size = 5                  # mini-batch size, not necessarily equal to the batch size in training
input_size = [572, 572]         # input size to NN, same as extracted patch size, no need to be the same as training
tile_size = [5000, 5000]        # size of the original image


tf.reset_default_graph()        # reset the graph before you start
# this is where I have my pretrained model
model_dir = r'/hdd/Models/UnetCrop_inria_aug_incity_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32'

# make collections
# same as what to do in training
blCol = uab_collectionFunctions.uabCollection('inria')
blCol.readMetadata()
file_list, parent_dir = blCol.getAllTileByDirAndExt([0, 1, 2])
file_list_truth, parent_dir_truth = blCol.getAllTileByDirAndExt(4)
idx, file_list = uabCrossValMaker.uabUtilGetFolds(None, file_list, 'force_tile')
idx_truth, file_list_truth = uabCrossValMaker.uabUtilGetFolds(None, file_list_truth, 'force_tile')
# use first 5 tiles for validation
file_list_valid = uabCrossValMaker.make_file_list_by_key(
    idx, file_list, [i for i in range(0, 6)],
    filter_list=['bellingham', 'bloomington', 'sfo', 'tyrol-e', 'innsbruck'])
file_list_valid_truth = uabCrossValMaker.make_file_list_by_key(
    idx_truth, file_list_truth, [i for i in range(0, 6)],
    filter_list=['bellingham', 'bloomington', 'sfo', 'tyrol-e', 'innsbruck'])
img_mean = blCol.getChannelMeans([0, 1, 2])

# make the model, same as training
X = tf.placeholder(tf.float32, shape=[None, input_size[0], input_size[1], 3], name='X')
y = tf.placeholder(tf.int32, shape=[None, input_size[0], input_size[1], 1], name='y')
mode = tf.placeholder(tf.bool, name='mode')
model = uabMakeNetwork_UNet.UnetModelCrop({'X':X, 'Y':y},
                                          trainable=mode,
                                          input_size=input_size,
                                          batch_size=5)
# create graph, same as training
model.create_graph('X', class_num=2)

# evaluate on tiles
# this is a wrapper function that helps you run on all images in file_list_valid defined above
# to evaluate on a single image, check examplescript_test_pretrained_model.ipynb
model.evaluate(file_list_valid,                 # list of lists, each inner list has all channel files names, can be
                                                # generated from uab_collection or just raw images
               file_list_valid_truth,           # list of lists, where each list is truth file
               parent_dir,                      # parent directory of where the images are stored
               parent_dir_truth,                # parent directory of where the truths are stroed
               input_size,                      # input size to NN, same as extracted patch size, no need to be the same
                                                # as training
               tile_size,                       # mini-batch size, not necessarily equal to the batch size in training
               batch_size,                      # mini-batch size, not necessarily equal to the batch size in training
               img_mean,                        # mean of rgb info
               model_dir,                       # path to pretrained model
               gpu,                             # which gpu to use
               save_result=True,                # if true, results will be saved in uabRepoPaths.evalPath in a folder by
                                                # its model name as a text file where each line is the IoU stats
               save_result_parent_dir=None,     # if not None, a folder will be create in uabRepoPaths.evalPath, this is
                                                # helpful when you score a bunch of models and make them more organized
               show_figure=False,               # if true, the prediction maps will be plotted after each iteration
               verb=True,                       # if true, the IoU and run duration will be printed
               ds_name='default',               # predictions maps will be saved under this folder
               load_epoch_num=None)             # load from a specific epoch
