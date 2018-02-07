import os
import numpy as np
import uabUtilreader
import util_functions


def get_tile_and_patch_num(chip_files):
    tile_name = [a[0].split('_')[0] for a in chip_files]
    tile_name = list(set(tile_name))
    tile_num = len(tile_name)
    patch_per_tile = int(len(chip_files)/tile_num)
    return tile_num, patch_per_tile, tile_name


def group_by_tile_name(tile_name, chip_files):
    group = [[] for i in range(len(tile_name))]
    tile_dict = {}
    for cnt, name in enumerate(tile_name):
        tile_dict[name] = cnt
    for item in chip_files:
        group[tile_dict[item[0].split('_')[0]]].append(item)
    return group


def group_by_city_name(tile_name, chip_files):
    '''group = [[] for i in range(len(tile_name))]
    tile_dict = {}
    for cnt, name in enumerate(tile_name):
        tile_dict[name[:3]] = cnt
    print(tile_dict)
    for item in chip_files:
        group[tile_dict[item[0].split('_')[0]][:3]].append(item)
    return group'''
    cities = list(set([a[:3] for a in tile_name]))
    city_dict = {}
    for cnt, name in enumerate(cities):
        city_dict[name] = cnt
    group = [[] for i in range(len(cities))]
    for item in chip_files:
        group[city_dict[item[0][:3]]].append(item)
    return group


# class to load all the possible slices of your data
class ImageLabelReader(object):
    def __init__(self, gtInds, dataInds, parentDir, chipFiles, chip_size, batchSize, nChannels=1,
                 padding=np.array((0, 0)), block_mean=None, dataAug='', batch_code=0):
        self.chip_size = chip_size
        self.block_mean = block_mean
        self.batch_code = batch_code

        # chipFiles:
        # list of lists.  Each inner list is a list of the chips by their extension.
        # These are all the input feature maps for a particular tile location
        # need to separate the file names into their own vectors

        if isinstance(chipFiles, str):
            filename = os.path.join(parentDir, chipFiles)
            with open(filename) as file:
                chipFiles = file.readlines()

            chipFiles = [a.strip().split(' ') for a in chipFiles]

        if nChannels is not list:
            self.nChannels = [nChannels for a in range(len(chipFiles[0]))]
        else:
            self.nChannels = nChannels

        el1 = chipFiles[0]
        if type(gtInds) is not list:
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

        fnameList = []
        for row in chipFiles:
            fnameList.append([row[i] for i in procInds])
        self.readManager = self.readFromDiskIteratorTrain(parentDir, fnameList, batchSize, self.chip_size, padding, dataAug)

    def readerAction(self, sess=None):
        return next(self.readManager)

    def readFromDiskIteratorTrain(self, image_dir, chipFiles, batch_size, patch_size, padding=(0, 0),
                                  dataAug=''):
        # this is a iterator for training
        if self.batch_code == 0:
            # pure random
            idx = np.random.permutation(len(chipFiles))
            nDims = len(chipFiles[0])
            while True:
                image_batch = np.zeros((batch_size, patch_size[0], patch_size[1], nDims))
                for cnt, randInd in enumerate(idx):
                    row = chipFiles[randInd]
                    blockList = []
                    nDims = 0
                    for file in row:
                        img = util_functions.uabUtilAllTypeLoad(os.path.join(image_dir, file))
                        if len(img.shape) == 2:
                            img = np.expand_dims(img, axis=2)
                        nDims += img.shape[2]
                        blockList.append(img)
                    block = np.dstack(blockList)

                    if self.block_mean is not None:
                        block -= self.block_mean

                    if dataAug != '':
                        augDat = uabUtilreader.doDataAug(block, nDims, dataAug)
                    else:
                        augDat = block

                    if (np.array(padding) > 0).any():
                        augDat = uabUtilreader.pad_block(augDat, padding)

                    image_batch[cnt % batch_size, :, :, :] = augDat

                    if ((cnt + 1) % batch_size == 0):
                        yield image_batch[:, :, : 1:], image_batch[:, :, :, :1]
        elif self.batch_code == 1:
            # random, batches from same tile
            tile_num, patch_per_tile, tile_name = get_tile_and_patch_num(chipFiles)
            group = group_by_tile_name(tile_name, chipFiles)

            tile_idx = np.random.permutation(tile_num)
            patch_idx = np.random.permutation(patch_per_tile)
            if patch_per_tile % batch_size != 0:
                comp_len = batch_size - patch_per_tile % batch_size
                patch_idx = np.append(patch_idx, patch_idx[:comp_len])
            nDims = len(chipFiles[0])
            while True:
                image_batch = np.zeros((batch_size, patch_size[0], patch_size[1], nDims))
                for randInd in tile_idx:
                    for cnt, patchInd in enumerate(patch_idx):
                        row = group[randInd][patchInd]
                        blockList = []
                        nDims = 0
                        for file in row:
                            img = util_functions.uabUtilAllTypeLoad(os.path.join(image_dir, file))
                            if len(img.shape) == 2:
                                img = np.expand_dims(img, axis=2)
                            nDims += img.shape[2]
                            blockList.append(img)
                        block = np.dstack(blockList)
                        block = block.astype(np.float32)

                        if self.block_mean is not None:
                            block -= self.block_mean

                        if dataAug != '':
                            augDat = uabUtilreader.doDataAug(block, nDims, dataAug, is_np=True)
                        else:
                            augDat = block

                        if (np.array(padding) > 0).any():
                            augDat = uabUtilreader.pad_block(augDat, padding)

                        image_batch[cnt % batch_size, :, :, :] = augDat

                        if ((cnt + 1) % batch_size == 0):
                            yield image_batch[:, :, :, 1:], image_batch[:, :, :, :1]
        else:
            # random, batches has to from different tiles
            tile_num, patch_per_tile, tile_name = get_tile_and_patch_num(chipFiles)
            group = group_by_city_name(tile_name, chipFiles)

            tile_idx = np.random.permutation(len(group))
            nDims = len(chipFiles[0])
            while True:
                image_batch = np.zeros((batch_size, patch_size[0], patch_size[1], nDims))
                for cnt, randInd in enumerate(tile_idx):
                    patchInd = np.random.randint(low=0, high=len(group[0]))
                    row = group[randInd][patchInd]
                    blockList = []
                    nDims = 0
                    for file in row:
                        img = util_functions.uabUtilAllTypeLoad(os.path.join(image_dir, file))
                        if len(img.shape) == 2:
                            img = np.expand_dims(img, axis=2)
                        nDims += img.shape[2]
                        blockList.append(img)
                    block = np.dstack(blockList)
                    block = block.astype(np.float32)

                    if self.block_mean is not None:
                        block -= self.block_mean

                    if dataAug != '':
                        augDat = uabUtilreader.doDataAug(block, nDims, dataAug, is_np=True)
                    else:
                        augDat = block

                    if (np.array(padding) > 0).any():
                        augDat = uabUtilreader.pad_block(augDat, padding)

                    image_batch[cnt % batch_size, :, :, :] = augDat

                    if ((cnt + 1) % batch_size == 0):
                        yield image_batch[:, :, :, 1:], image_batch[:, :, :, :1]


# for debugging purposes
if __name__ == '__main__':
    import uab_collectionFunctions
    import bohaoCustom.uabPreprocClasses as bPreproc
    import uab_DataHandlerFunctions
    import uabCrossValMaker

    # create collection
    # the original file is in /ei-edl01/data/uab_datasets/inria
    blCol = uab_collectionFunctions.uabCollection('inria')
    img_mean = blCol.getChannelMeans([0, 1, 2])

    # extract patches
    extrObj = uab_DataHandlerFunctions.uabPatchExtr([0, 1, 2, 4],  # extract all 4 channels
                                                    cSize=(572, 572),  # patch size as 572*572
                                                    numPixOverlap=int(184 / 2),  # overlap as 92
                                                    extSave=['jpg', 'jpg', 'jpg', 'png'],
                                                    # save rgb files as jpg and gt as png
                                                    isTrain=True,
                                                    gtInd=3,
                                                    pad=184)  # pad around the tiles
    patchDir = extrObj.run(blCol)

    # make data reader
    chipFiles = os.path.join(patchDir, 'fileList.txt')
    # use uabCrossValMaker to get fileLists for training and validation
    idx, file_list = uabCrossValMaker.uabUtilGetFolds(patchDir, 'fileList.txt', 'force_tile')
    # use first 5 tiles for validation
    file_list_train = uabCrossValMaker.make_file_list_by_key(idx, file_list, [i for i in range(6, 37)])
    file_list_valid = uabCrossValMaker.make_file_list_by_key(idx, file_list, [i for i in range(0, 6)])

    dataReader_train = ImageLabelReader([3], [0, 1, 2], patchDir, file_list_train, (572, 572),
                                        5, dataAug='flip,rotate', block_mean=np.append([0], img_mean),
                                        batch_code=2)

    for plt_cnt in range(10):
        x, y = dataReader_train.readerAction()
        import matplotlib.pyplot as plt
        for i in range(5):
            plt.subplot(5, 2, i*2+1)
            plt.imshow(x[i, :, :, :]+img_mean)
            plt.subplot(5, 2, i*2+1+1)
            plt.imshow(y[i, :, :, 0])
        plt.show()
