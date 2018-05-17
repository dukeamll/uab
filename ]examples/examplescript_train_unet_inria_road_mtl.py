"""
Train a U-Net model on both Inria building and MIT road dataset
The model has one encoder with weights shared and two seperate decoder, one for each dataset

@author: Bohao
"""

import time
import argparse
import numpy as np
import tensorflow as tf
import uabDataReader
import uabRepoPaths
import uabCrossValMaker
import bohaoCustom.uabPreprocClasses as bPreproc
import uabPreprocClasses
import uab_collectionFunctions
import uab_DataHandlerFunctions
from bohaoCustom import uabMakeNetwork_UnetMTL

RUN_ID = 0
BATCH_SIZE = 5
LEARNING_RATE = 1e-4
INPUT_SIZE = 572
TILE_SIZE = 5000
EPOCHS = 100
NUM_CLASS = '2,2'                       # both datasets have binary labels
N_TRAIN = 8000
N_VALID = 1000
GPU = 0
DECAY_STEP = 60
DECAY_RATE = 0.1
MODEL_NAME = 'inria_road_mtl_{}'
SFN = 32
S_NUM = 2                               # number of input sources
S_NAME = 'INRIA,ROAD'                   # name of input sources, these will affect tile in tensorboard
S_CONTROL = '2,1'                       # load inria data twice and road data once per step


def read_flag():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=BATCH_SIZE, type=int, help='batch size (10)')
    parser.add_argument('--learning-rate', type=float, default=LEARNING_RATE, help='learning rate (1e-3)')
    parser.add_argument('--input-size', default=INPUT_SIZE, type=int, help='input size 224')
    parser.add_argument('--tile-size', default=TILE_SIZE, type=int, help='tile size 5000')
    parser.add_argument('--epochs', default=EPOCHS, type=int, help='# epochs (1)')
    parser.add_argument('--num-classes', type=str, default=NUM_CLASS, help='# classes (including background)')
    parser.add_argument('--n-train', type=int, default=N_TRAIN, help='# samples per epoch')
    parser.add_argument('--n-valid', type=int, default=N_VALID, help='# patches to valid')
    parser.add_argument('--GPU', type=str, default=GPU, help="GPU used for computation.")
    parser.add_argument('--decay-step', type=float, default=DECAY_STEP, help='Learning rate decay step in number of epochs.')
    parser.add_argument('--decay-rate', type=float, default=DECAY_RATE, help='Learning rate decay rate')
    parser.add_argument('--model-name', type=str, default=MODEL_NAME, help='Model name')
    parser.add_argument('--run-id', type=str, default=RUN_ID, help='id of this run')
    parser.add_argument('--sfn', type=int, default=SFN, help='filter number of the first layer')
    parser.add_argument('--s-num', type=int, default=S_NUM, help='number of input sources')
    parser.add_argument('--s-name', type=str, default=S_NAME, help='names of input sources')
    parser.add_argument('--s-control', type=str, default=S_CONTROL, help='control portions of input sources')

    flags = parser.parse_args()
    flags.input_size = (flags.input_size, flags.input_size)
    flags.tile_size = (flags.tile_size, flags.tile_size)
    flags.model_name = flags.model_name.format(flags.run_id)
    flags.s_name = flags.s_name.split(',')
    flags.s_control = [int(i) for i in flags.s_control.split(',')]
    flags.num_classes = [int(i) for i in flags.num_classes.split(',')]
    return flags


def main(flags):
    # ------------------------------------------Network---------------------------------------------#
    # make network
    # define place holder
    X = tf.placeholder(tf.float32, shape=[None, flags.input_size[0], flags.input_size[1], 3], name='X')
    y = tf.placeholder(tf.int32, shape=[None, flags.input_size[0], flags.input_size[1], 1], name='y')
    mode = tf.placeholder(tf.bool, name='mode')
    model = uabMakeNetwork_UnetMTL.UnetModelMTL({'X':X, 'Y':y},
                                                trainable=mode,
                                                model_name=flags.model_name,
                                                input_size=flags.input_size,
                                                batch_size=flags.batch_size,
                                                learn_rate=flags.learning_rate,
                                                decay_step=flags.decay_step,
                                                decay_rate=flags.decay_rate,
                                                epochs=flags.epochs,
                                                start_filter_num=flags.sfn,
                                                source_num=flags.s_num,
                                                source_name=flags.s_name,
                                                source_control=flags.s_control)
    model.create_graph('X', class_num=flags.num_classes, start_filter_num=flags.sfn)

    # ------------------------------------------Dataset Inria---------------------------------------------#
    # create collection for inria
    blCol_inria = uab_collectionFunctions.uabCollection('inria')
    opDetObj_inria = bPreproc.uabOperTileDivide(255)
    # [3] is the channel id of GT
    rescObj = uabPreprocClasses.uabPreprocMultChanOp([], 'GT_Divide.tif', 'Map GT to (0, 1)', [3], opDetObj_inria)
    rescObj.run(blCol_inria)
    img_mean_inria = blCol_inria.getChannelMeans([0, 1, 2])

    # extract patches
    extrObj_inria = uab_DataHandlerFunctions.uabPatchExtr([0, 1, 2, 4],
                                                          cSize=flags.input_size,
                                                          numPixOverlap=int(model.get_overlap()),
                                                          extSave=['jpg', 'jpg', 'jpg', 'png'],
                                                          isTrain=True,
                                                          gtInd=3,
                                                          pad=int(model.get_overlap() / 2))
    patchDir_inria = extrObj_inria.run(blCol_inria)

    # make data reader
    # use uabCrossValMaker to get fileLists for training and validation
    idx_inria, file_list_inria = uabCrossValMaker.uabUtilGetFolds(patchDir_inria, 'fileList.txt', 'force_tile')
    # use first 5 tiles for validation
    file_list_train_inria = uabCrossValMaker.make_file_list_by_key(idx_inria, file_list_inria,
                                                                   [i for i in range(20, 136)])
    file_list_valid_inria = uabCrossValMaker.make_file_list_by_key(idx_inria, file_list_inria,
                                                                   [i for i in range(0, 20)])

    with tf.name_scope('image_loader_inria'):
        # GT has no mean to subtract, append a 0 for block mean
        dataReader_train_inria = uabDataReader.ImageLabelReader([3], [0, 1, 2], patchDir_inria, file_list_train_inria,
                                                                flags.input_size, flags.tile_size, flags.batch_size,
                                                                dataAug='flip,rotate',
                                                                block_mean=np.append([0], img_mean_inria))
        # no augmentation needed for validation
        dataReader_valid_inria = uabDataReader.ImageLabelReader([3], [0, 1, 2], patchDir_inria, file_list_valid_inria,
                                                                flags.input_size, flags.tile_size,
                                                                flags.batch_size, dataAug=' ',
                                                                block_mean=np.append([0], img_mean_inria))

    # ------------------------------------------Dataset Road---------------------------------------------#
    # create collection for road
    blCol_road = uab_collectionFunctions.uabCollection('road_5000')
    opDetObj_road = bPreproc.uabOperTileDivide(255)
    # [3] is the channel id of GT
    rescObj = uabPreprocClasses.uabPreprocMultChanOp([], 'GT_Divide.tif', 'Map GT to (0, 1)', [3], opDetObj_road)
    rescObj.run(blCol_road)
    img_mean_road = blCol_road.getChannelMeans([0, 1, 2])

    # extract patches
    extrObj_road = uab_DataHandlerFunctions.uabPatchExtr([0, 1, 2, 4],
                                                         cSize=flags.input_size,
                                                         numPixOverlap=int(model.get_overlap()),
                                                         extSave=['jpg', 'jpg', 'jpg', 'png'],
                                                         isTrain=True,
                                                         gtInd=3,
                                                         pad=int(model.get_overlap() / 2))
    patchDir_road = extrObj_road.run(blCol_road)

    # make data reader
    # use uabCrossValMaker to get fileLists for training and validation
    idx_road, file_list_road = uabCrossValMaker.uabUtilGetFolds(patchDir_road, 'fileList.txt', 'city')
    # use first 5 tiles for validation
    file_list_train_road = uabCrossValMaker.make_file_list_by_key(idx_road, file_list_road, [1])
    file_list_valid_road = uabCrossValMaker.make_file_list_by_key(idx_road, file_list_road, [0, 2])

    with tf.name_scope('image_loader_road'):
        # GT has no mean to subtract, append a 0 for block mean
        dataReader_train_road = uabDataReader.ImageLabelReader([3], [0, 1, 2], patchDir_road, file_list_train_road,
                                                                flags.input_size, flags.tile_size, flags.batch_size,
                                                                dataAug='flip,rotate',
                                                                block_mean=np.append([0], img_mean_road))
        # no augmentation needed for validation
        dataReader_valid_road = uabDataReader.ImageLabelReader([3], [0, 1, 2], patchDir_road, file_list_valid_road,
                                                                flags.input_size, flags.tile_size,
                                                                flags.batch_size, dataAug=' ',
                                                                block_mean=np.append([0], img_mean_road))

    # ------------------------------------------Train---------------------------------------------#
    start_time = time.time()

    model.train_config('X', 'Y', flags.n_train, flags.n_valid, flags.input_size, uabRepoPaths.modelPath,
                       loss_type='xent')
    model.run(train_reader=[dataReader_train_inria, dataReader_train_road],
              valid_reader=[dataReader_valid_inria, dataReader_valid_road],
              pretrained_model_dir=None,        # train from scratch, no need to load pre-trained model
              isTrain=True,
              img_mean=[img_mean_inria, img_mean_road],
              verb_step=100,                    # print a message every 100 step(sample)
              save_epoch=5,                     # save the model every 5 epochs
              gpu=GPU,
              tile_size=flags.tile_size,
              patch_size=flags.input_size)

    duration = time.time() - start_time
    print('duration {:.2f} hours'.format(duration/60/60))


if __name__ == '__main__':
    flags = read_flag()
    main(flags)
