import os
import imageio
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from shutil import rmtree
from shutil import copyfile
import util_functions
import uabBlockparent
import uab_DataHandlerFunctions
from uabBlockparent import uabBlock


def compute_class_percentage(gt, class_label=1):
    return np.sum(gt == class_label) / (gt.shape[0] * gt.shape[1])


class uabPatchExtrClassSelect(uab_DataHandlerFunctions.uabPatchExtr):
    fname = 'fileList.txt'

    def __init__(self, runChannels, name='RegSelect', cSize=(224, 224), numPixOverlap=0, pad=0, extSave=None,
                 isTrain=False, gtInd=-1, class_label=1, select_percent=0.4):
        super(uabPatchExtrClassSelect, self).__init__(runChannels, name)
        # chip size
        self.chipExtrSize = cSize
        # numPixOverlap -> number of pixels to overlap the patches by.  If = 0, then extract tiles such that the first patch starts on row 1 and the final patch ends on the final row (& dito for columns)
        self.numPixOverlap = numPixOverlap
        self.pad = pad
        self.coord = tf.train.Coordinator()
        self.saveExts = extSave
        self.isTrain = isTrain
        self.gtInd = gtInd
        self.class_label = class_label
        self.select_percent = select_percent

    def runAction(self, colObj):
        # function to extract the chips from the tiles

        gridList = self.makeGrid([colObj.tileSize[0] + self.pad, colObj.tileSize[1] + self.pad])

        directory = self.getDirectoryPaths(colObj)
        # check if regular directory exists
        read_from_old = False
        if os.path.exists(directory.replace('Select', '')):
            directory = directory.replace('Select', '')
            read_from_old = True
        else:
            # extract chips for all the specified extensions
            # precompute extensions
            fileExts = []
            for cnt, chanId in enumerate(self.runChannels):
                ext, _ = colObj.getExtensionInfoById(chanId)
                if (self.saveExts is not None):
                    sExt = ext.split('.')
                    fileExts.append(sExt[0] + '.' + self.saveExts[cnt])
                else:
                    fileExts.append(ext)

            f_temp = []
            for i in range(len(fileExts)):
                f_temp.append([])
            for ind, tilename in enumerate(tqdm(colObj.dataListForRun)):
                if self.isTrain:
                    # check if gt exists for this tile
                    try:
                        colObj.loadTileDataByExtension(tilename, self.runChannels[self.gtInd])
                    except IOError:
                        # skip this if there's not enough channels
                        continue
                for cnt, (ext, chanId) in enumerate(zip(fileExts, self.runChannels)):
                    cIm = colObj.loadTileDataByExtension(tilename, chanId)
                    if self.pad > 0:
                        cIm = np.pad(cIm, ((self.pad, self.pad), (self.pad, self.pad)), 'symmetric')
                    nDims = cIm.shape
                    for coordList in gridList:
                        # extract patches for all the channels at coordinate location.
                        # This is done so that the file containing patch names can have all
                        # the extracted patches of one location on a single line
                        x1 = int(coordList[0])
                        x2 = int(coordList[1])
                        finNm = tilename + '_y%dx%d_%s' % (x1, x2, ext)

                        fPath = os.path.join(directory, finNm)
                        isExt = util_functions.read_or_new_pickle(fPath, toLoad=0)
                        if (isExt == 0):
                            # extract a patch from the image
                            if (len(nDims) == 2):
                                chipDat = cIm[x1:x1 + self.chipExtrSize[0], x2:x2 + self.chipExtrSize[1]]
                            else:
                                chipDat = cIm[x1:x1 + self.chipExtrSize[0], x2:x2 + self.chipExtrSize[1], :]

                            util_functions.read_or_new_pickle(fPath, toSave=1, variable_to_save=chipDat)

                        f_temp[cnt].append(finNm)

                with open(os.path.join(directory, uabPatchExtrClassSelect.fname), 'w') as file:
                    for i in range(len(f_temp[0])):
                        s = []
                        for j in range(len(f_temp)):
                            s.append(f_temp[j][i])
                        file.write('{}\n'.format(' '.join(s)))

        # make new dir
        if read_from_old:
            directory_new = uabBlock.getBlockDir(os.path.join(uabBlockparent.outputDirs['patchExt'], colObj.colName,
                                                              self.algoName()))
        else:
            directory_new = uabBlock.getBlockDir(os.path.join(uabBlockparent.outputDirs['patchExt'], colObj.colName,
                                                          self.algoName() + 'Select'))
        if not os.path.exists(directory_new):
            os.makedirs(directory_new)

        files = os.path.join(directory, 'fileList.txt')
        with open(files, 'r') as f:
            file_list = f.readlines()
        file_list_new = []

        print('Selecting Rule:\n\tLabel Class: {}, Select Percentage: {}%'.
              format(self.class_label, int(self.select_percent * 100)))
        for file in tqdm(file_list):
            f_array = file.strip().split(' ')
            gt = imageio.imread(os.path.join(directory, f_array[-1]))
            class_percent = compute_class_percentage(gt, self.class_label)

            if class_percent > self.select_percent:
                file_list_new.append(file)
                for i in f_array[:3]:
                    copyfile(os.path.join(directory, i), os.path.join(directory_new, i))
                    copyfile(os.path.join(directory, f_array[-1]), os.path.join(directory_new, f_array[-1]))

        files = os.path.join(directory_new, 'fileList.txt')
        with open(files, 'w+') as f:
            for file in file_list_new:
                f.write(file)

        # remove old directory
        if not read_from_old:
            rmtree(directory)

        files = os.path.join(directory_new, 'state.txt')
        with open(files, 'w+') as f:
            f.write('Finished\n')

        # rename directory
        if not read_from_old:
            os.rename(directory_new, directory)

        old_patch_num = len(file_list)
        new_patch_num = len(file_list_new)
        print('After selection, {:.2f}% patches kept'.format(new_patch_num / old_patch_num * 100))
