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
                    block = np.dstack(blockList).astype(np.float32)

                    if self.block_mean is not None:
                        block -= self.block_mean

                    if dataAug != '':
                        augDat = uabUtilreader.doDataAug(block, nDims, dataAug, is_np=True, img_mean=self.block_mean)
                    else:
                        augDat = block

                    if (np.array(padding) > 0).any():
                        augDat = uabUtilreader.pad_block(augDat, padding)

                    image_batch[cnt % batch_size, :, :, :] = augDat

                    if ((cnt + 1) % batch_size == 0):
                        yield image_batch[:, :, :, 1:], image_batch[:, :, :, :1]
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
                        block = np.dstack(blockList).astype(np.float32)

                        if self.block_mean is not None:
                            block -= self.block_mean

                        if dataAug != '':
                            augDat = uabUtilreader.doDataAug(block, nDims, dataAug, is_np=True, img_mean=self.block_mean)
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
                    block = np.dstack(blockList).astype(np.float32)

                    if self.block_mean is not None:
                        block -= self.block_mean

                    if dataAug != '':
                        augDat = uabUtilreader.doDataAug(block, nDims, dataAug, is_np=True, img_mean=self.block_mean)
                    else:
                        augDat = block

                    if (np.array(padding) > 0).any():
                        augDat = uabUtilreader.pad_block(augDat, padding)

                    image_batch[cnt % batch_size, :, :, :] = augDat

                    if ((cnt + 1) % batch_size == 0):
                        yield image_batch[:, :, :, 1:], image_batch[:, :, :, :1]

# class to load all the possible slices of your data
class ImageLabelReader_City(object):
    def __init__(self, gtInds, dataInds, parentDir, chipFiles, chip_size, batchSize, city_dict, nChannels=1,
                 padding=np.array((0, 0)), block_mean=None, dataAug=''):
        self.chip_size = chip_size
        self.block_mean = block_mean
        self.city_dict = city_dict

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
        self.readManager = self.readFromDiskIteratorTrain(parentDir, fnameList, batchSize,
                                                          self.chip_size, padding, dataAug)

    def readerAction(self, sess=None):
        return next(self.readManager)

    def readFromDiskIteratorTrain(self, image_dir, chipFiles, batch_size, patch_size,
                                  padding=(0, 0), dataAug=''):
        # this is a iterator for training
        # pure random
        idx = np.random.permutation(len(chipFiles))
        nDims = len(chipFiles[0])
        while True:
            image_batch = np.zeros((batch_size, patch_size[0], patch_size[1], nDims))
            cityid_batch = np.zeros(batch_size, dtype=np.uint8)
            for cnt, randInd in enumerate(idx):
                row = chipFiles[randInd]
                blockList = []
                nDims = 0
                city_name = ''.join([a for a in row[0].split('_')[0] if not a.isdigit()])
                cityid_batch[cnt%batch_size] = self.city_dict[city_name]
                for file in row:
                    img = util_functions.uabUtilAllTypeLoad(os.path.join(image_dir, file))
                    if len(img.shape) == 2:
                        img = np.expand_dims(img, axis=2)
                    nDims += img.shape[2]
                    blockList.append(img)
                block = np.dstack(blockList).astype(np.float32)

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
                    yield image_batch[:, :, :, 1:], image_batch[:, :, :, :1], cityid_batch


class ImageLabelReaderCitySampleControl(ImageLabelReader):
    def __init__(self, gtInds, dataInds, parentDir, chipFiles, chip_size, batchSize, city_dict, city_alpha,
                 nChannels=1, padding=np.array((0, 0)), block_mean=None, dataAug=''):
        self.city_dict = city_dict
        self.city_alpha = city_alpha
        super(ImageLabelReaderCitySampleControl, self).__init__(gtInds, dataInds, parentDir, chipFiles, chip_size,
                                                                   batchSize, nChannels, padding, block_mean, dataAug)

    def get_group_city_sorted_alpha(self, group):
        alpha = []
        for i in range(len(group)):
            city_name = ''.join([c for c in group[i][0][0].split('_')[0] if not c.isdigit()])
            alpha.append(self.city_alpha[self.city_dict[city_name]])
        return alpha

    def readFromDiskIteratorTrain(self, image_dir, chipFiles, batch_size, patch_size, padding=(0, 0),
                                  dataAug=''):
        # this is a iterator for training
        tile_num, patch_per_tile, tile_name = get_tile_and_patch_num(chipFiles)
        group = group_by_city_name(tile_name, chipFiles)
        assert len(group) == len(self.city_alpha)
        alpha = self.get_group_city_sorted_alpha(group)
        random_id = [np.random.permutation(len(group[i])) for i in range(len(group))]
        group_cnt = [0 for i in range(len(group))]

        nDims = len(chipFiles[0])
        while True:
            image_batch = np.zeros((batch_size, patch_size[0], patch_size[1], nDims))
            # select number to sample
            city_batch = np.random.choice(len(group), batch_size, p=alpha)
            for cnt, randInd in enumerate(city_batch):
                patchInd = random_id[randInd][group_cnt[randInd] % len(group[randInd])]
                group_cnt[randInd] += 1
                row = group[randInd][patchInd]
                blockList = []
                nDims = 0
                for file in row:
                    img = util_functions.uabUtilAllTypeLoad(os.path.join(image_dir, file))
                    if len(img.shape) == 2:
                        img = np.expand_dims(img, axis=2)
                    nDims += img.shape[2]
                    blockList.append(img)
                block = np.dstack(blockList).astype(np.float32)

                if self.block_mean is not None:
                    block -= self.block_mean

                if dataAug != '':
                    augDat = uabUtilreader.doDataAug(block, nDims, dataAug, is_np=True, img_mean=self.block_mean)
                else:
                    augDat = block

                if (np.array(padding) > 0).any():
                    augDat = uabUtilreader.pad_block(augDat, padding)

                image_batch[cnt % batch_size, :, :, :] = augDat

                if ((cnt + 1) % batch_size == 0):
                    yield image_batch[:, :, :, 1:], image_batch[:, :, :, :1]

class ImageLabelReaderGroupSampleControl(ImageLabelReader):
    def __init__(self, gtInds, dataInds, parentDir, chipFiles, chip_size, batchSize,
                 group_alpha, group_files,
                 nChannels=1, padding=np.array((0, 0)), block_mean=None, dataAug=''):
        self.group_alpha = group_alpha
        self.group_files = group_files
        super(ImageLabelReaderGroupSampleControl, self).__init__(gtInds, dataInds, parentDir, chipFiles,
                                                                chip_size,
                                                                batchSize, nChannels, padding,
                                                                block_mean, dataAug)

    def group_chip_files(self, chipFiles):
        group_num = len(self.group_files)
        group = [[] for i in range(group_num)]
        for item in chipFiles:
            name_id = item[0].split('_')[0]
            for i in range(group_num):
                if name_id in self.group_files[i]:
                    group[i].append(item)
                    break
        return group

    def readFromDiskIteratorTrain(self, image_dir, chipFiles, batch_size, patch_size, padding=(0, 0),
                                  dataAug=''):
        # this is a iterator for training
        group = self.group_chip_files(chipFiles)
        assert len(group) == len(self.group_alpha)
        random_id = [np.random.permutation(len(group[i])) for i in range(len(group))]
        group_cnt = [0 for i in range(len(group))]

        nDims = len(chipFiles[0])
        while True:
            image_batch = np.zeros((batch_size, patch_size[0], patch_size[1], nDims))
            # select number to sample
            city_batch = np.random.choice(len(group), batch_size, p=self.group_alpha)
            for cnt, randInd in enumerate(city_batch):
                patchInd = random_id[randInd][group_cnt[randInd] % len(group[randInd])]
                group_cnt[randInd] += 1
                row = group[randInd][patchInd]
                blockList = []
                nDims = 0
                for file in row:
                    img = util_functions.uabUtilAllTypeLoad(os.path.join(image_dir, file))
                    if len(img.shape) == 2:
                        img = np.expand_dims(img, axis=2)
                    nDims += img.shape[2]
                    blockList.append(img)
                block = np.dstack(blockList).astype(np.float32)

                if self.block_mean is not None:
                    block -= self.block_mean

                if dataAug != '':
                    augDat = uabUtilreader.doDataAug(block, nDims, dataAug, is_np=True,
                                                     img_mean=self.block_mean)
                else:
                    augDat = block

                if (np.array(padding) > 0).any():
                    augDat = uabUtilreader.pad_block(augDat, padding)

                image_batch[cnt % batch_size, :, :, :] = augDat

                if ((cnt + 1) % batch_size == 0):
                    yield image_batch[:, :, :, 1:], image_batch[:, :, :, :1]

class ImageLabelReaderPatchSampleControl(ImageLabelReader):
    def __init__(self, gtInds, dataInds, parentDir, chipFiles, chip_size, batchSize,
                 patch_prob, patch_name=False,
                 nChannels=1, padding=np.array((0, 0)), block_mean=None, dataAug=''):
        self.patch_prob = patch_prob
        self.return_name = patch_name
        super(ImageLabelReaderPatchSampleControl, self).__init__(gtInds, dataInds, parentDir, chipFiles,
                                                                 chip_size,
                                                                 batchSize, nChannels, padding,
                                                                 block_mean, dataAug)

    def readFromDiskIteratorTrain(self, image_dir, chipFiles, batch_size, patch_size, padding=(0, 0),
                                  dataAug=''):
        # this is a iterator for training
        nDims = len(chipFiles[0])
        assert len(chipFiles) == len(self.patch_prob)
        while True:
            image_batch = np.zeros((batch_size, patch_size[0], patch_size[1], nDims))
            patch_name = [[] for i in range(batch_size)]
            # select number to sample
            idx_batch = np.random.choice(len(chipFiles), batch_size, p=self.patch_prob)
            for cnt, randInd in enumerate(idx_batch):
                row = chipFiles[randInd]
                p_name = '_'.join(row[0].split('_')[:2])

                blockList = []
                nDims = 0
                for file in row:
                    img = util_functions.uabUtilAllTypeLoad(os.path.join(image_dir, file))
                    if len(img.shape) == 2:
                        img = np.expand_dims(img, axis=2)
                    nDims += img.shape[2]
                    blockList.append(img)
                block = np.dstack(blockList).astype(np.float32)

                if self.block_mean is not None:
                    block -= self.block_mean

                if dataAug != '':
                    augDat = uabUtilreader.doDataAug(block, nDims, dataAug, is_np=True,
                                                     img_mean=self.block_mean)
                else:
                    augDat = block

                if (np.array(padding) > 0).any():
                    augDat = uabUtilreader.pad_block(augDat, padding)

                store_idx = cnt % batch_size
                image_batch[store_idx, :, :, :] = augDat
                patch_name[store_idx] = p_name

                if (cnt + 1) % batch_size == 0:
                    if self.return_name:
                        yield image_batch[:, :, :, 1:], image_batch[:, :, :, :1], patch_name
                    else:
                        yield image_batch[:, :, :, 1:], image_batch[:, :, :, :1]

class ImageLabelReaderBuilding(ImageLabelReader):
    def __init__(self, gtInds, dataInds, parentDir, chipFiles, chip_size, batchSize, patch_prob,
                 nChannels=1, padding=np.array((0, 0)), block_mean=None, dataAug=''):
        self.patch_prob = patch_prob
        super(ImageLabelReaderBuilding, self).__init__(gtInds, dataInds, parentDir,
                                                       chipFiles,
                                                       chip_size,
                                                       batchSize, nChannels, padding,
                                                       block_mean, dataAug)

    def readFromDiskIteratorTrain(self, image_dir, chipFiles, batch_size, patch_size,
                                  padding=(0, 0), dataAug=''):
        # this is a iterator for training
        nDims = len(chipFiles[0])
        while True:
            image_batch = np.zeros((batch_size, patch_size[0], patch_size[1], nDims))
            building_truth = np.zeros((batch_size, 1))
            # select number to sample
            idx_batch = np.random.permutation(len(chipFiles))
            for cnt, randInd in enumerate(idx_batch):
                row = chipFiles[randInd]
                p_name = '_'.join(row[0].split('_')[:2])

                blockList = []
                nDims = 0
                for file in row:
                    img = util_functions.uabUtilAllTypeLoad(os.path.join(image_dir, file))
                    if len(img.shape) == 2:
                        img = np.expand_dims(img, axis=2)
                    nDims += img.shape[2]
                    blockList.append(img)
                block = np.dstack(blockList).astype(np.float32)

                if self.block_mean is not None:
                    block -= self.block_mean

                if dataAug != '':
                    augDat = uabUtilreader.doDataAug(block, nDims, dataAug, is_np=True,
                                                     img_mean=self.block_mean)
                else:
                    augDat = block

                if (np.array(padding) > 0).any():
                    augDat = uabUtilreader.pad_block(augDat, padding)

                store_idx = cnt % batch_size
                image_batch[store_idx, :, :, :] = augDat
                if np.sum(image_batch[store_idx, :, :, 0]) / (patch_size[0] * patch_size[0]) > self.patch_prob:
                    building_truth[store_idx, :] = 1
                else:
                    building_truth[store_idx, :] = 0

                if (cnt + 1) % batch_size == 0:
                    yield image_batch[:, :, :, 1:], image_batch[:, :, :, :1], building_truth


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
                                                    cSize=(321, 321),  # patch size as 572*572
                                                    numPixOverlap=0,  # overlap as 92
                                                    extSave=['jpg', 'jpg', 'jpg', 'png'],
                                                    # save rgb files as jpg and gt as png
                                                    isTrain=True,
                                                    gtInd=3,
                                                    pad=0)  # pad around the tiles
    patchDir = extrObj.run(blCol)

    # make data reader
    chipFiles = os.path.join(patchDir, 'fileList.txt')
    # use uabCrossValMaker to get fileLists for training and validation
    idx, file_list = uabCrossValMaker.uabUtilGetFolds(patchDir, 'fileList.txt', 'force_tile')
    # use first 5 tiles for validation
    file_list_train = uabCrossValMaker.make_file_list_by_key(idx, file_list, [i for i in range(6, 37)])
    file_list_valid = uabCrossValMaker.make_file_list_by_key(idx, file_list, [i for i in range(0, 6)])

    dataReader_train = ImageLabelReaderBuilding([3], [0, 1, 2], patchDir, file_list_train, (321, 321),
                                                5, 0.1, block_mean=np.append([0], img_mean))

    for plt_cnt in range(10):
        x, y, b = dataReader_train.readerAction()
        print(b)
        import matplotlib.pyplot as plt
        for i in range(5):
            plt.subplot(5, 2, i*2+1)
            plt.imshow((x[i, :, :, :]+img_mean).astype(np.uint8))
            plt.subplot(5, 2, i*2+1+1)
            plt.imshow(y[i, :, :, 0])
        plt.show()

    '''idx_all = np.zeros(50000)
    for plt_cnt in range(10000):
        _, _, idx = dataReader_train.readerAction()
        idx_all[plt_cnt*5:(plt_cnt+1)*5] = idx
    import matplotlib.pyplot as plt
    plt.hist(idx_all)
    plt.show()'''
