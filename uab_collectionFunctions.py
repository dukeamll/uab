# -*- coding: utf-8 -*-
"""
Created on Mon Nov 06 21:25:52 2017

@author: Daniel

Dataset handler.  All tile-level transformations & meta-data are saved on the server.  Results of patch extraction etc. are all (expected to be) saved locally.  This is done to save some space across users of the same dataset.  Saving patches locally is done to speed up data-access during training

This class contains functions to keep the accounting of the tiles & their associated files.  A new collection should be organized in the format specified below.  This is manual labor someone will need to perform :).  In this file, the default names are written into the class.  You do not need to make the "Processed_Tiles" folders.

Default behavior: to simplify accounting later down the line, RGB channels are divorced and saved in separate files.  Each file contains a single channel.  At collection initialization, a check is performed whether this has been performed.  This will happen before any other tile-level operations are performed.

server/
    path/to/data/[Dataset Name 1]/
        data/
            Original_Tiles/
                All the files associated to all of the tiles as downloaded from another place.  This includes all data (even if some files don't have ground truth associated to them) and all the ground truth (even if such files are missing for some of the data files)
                Each tile can have several channels (e.g., RGB, infrared, height).  All files associated to a particular tile should be differ by their postfix and extension ONLY.  Separate parts of the tilename using underscores ('_').  An example is shown here:
                    TileName_CityName_RGB.tif
                    TileName_CityName_DSM.tif
            Processed_Tiles/
                [Directory for all preprocessed tiles of this dataset, organized by folder]
                preproc_result1/
                    [In the preproc object, both a postfix & an extension are specified to append to the tilename during this action]
                    TileName_preprocExtension.extension
                preproc_result2/
        meta_data/
            collection.txt
                file that is updated each time a new channel is made using preprocessing
            colTileNames.txt
                file that contains the name of each tile in the collection without extensions
            mean_values.npy
                file that contains the mean value of each channel
        
        collectionMeta.txt
            a user-made file that specifies information of interest about this collection (e.g., data-resolution)
    path/to/data/[Dataset Name 2]/
    path/to/data/[Dataset Name 3]/
    
    
Future work:
    (1) setExtensions(): Automatically handle pruning the list of processed tiles in the metadata if some were deleted by transferring the descriptions of previously processed tiles 
"""


import os, glob, pickle
import numpy as np
from tqdm import tqdm
import uabRepoPaths
import util_functions, uabPreprocClasses

class uabCollection(object):
    
    #These are sub-directories to the collection name as specified above
    colDirnames = {'orig': 'Original_Tiles', 'proc' : 'Processed_Tiles'}
    dataDirnames = {'data':'data', 'meta':'meta_data'}
    
    metaFilename = 'collection.txt'
    tileNamesFile = 'colTileNames.txt'
    metaInfoFile = 'meta.npy'                       # file to store all meta information
    
    def __init__(self, colN, splitChans = 1):
        #full path to data directory
        self.colName = colN
        self.imDirectory = os.path.join(uabRepoPaths.dataPath, self.colName)
        
        #make the meta-data directory
        util_functions.uabUtilMakeDirectoryName(os.path.join(self.imDirectory, uabCollection.dataDirnames['meta']))
        
        #list of the names of tiles for this collection.  DO NOT MODIFY
        self.tileList = self.getImLists('orig')
        
        #For a block to know on which data to run, we provide either:
        #(1) a list of tiles
        #(2) the path to a file with a list of files 
        #depending on whether this is a tile-level or patch-level block.  Blocks will modify this property.  DO NOT MODIFY TILELIST
        self.dataListForRun = self.tileList
        
        #data structure that associates the tile channels to a number that the user can select
        self.extDS = []
        #make mapping of processed tiles to extensions.  By default we should have RGB split up- check if this has happened otherwise, do it
        self.setExtensions(splitChans)
        
        #get tile size
        im = self.loadTileDataByExtension(self.tileList[0], 0)
        self.tileSize = im.shape
        
        #get tile tile-mean
        #self.colMean = self.computeTrainingMean()
        
        print('Warning: do not forget to select tile-types for patch extraction.  Run function readMetadata() to obtain tile-level information')
    
    def getImLists(self, colDir = 'orig', forcerun = 0):
        #returns the name of all the tiles in the dataDirectory.  Removes extensions
        colFileName = os.path.join(self.imDirectory, uabCollection.dataDirnames['meta'], uabCollection.tileNamesFile)
        if(os.path.exists(colFileName) or forcerun == 1):
            with open(colFileName, 'r') as file:
                return [a.strip() for a in file.readlines()]
        else:
            filenames = glob.glob(os.path.join(self.imDirectory, uabCollection.dataDirnames['data'], uabCollection.colDirnames[colDir], '*.*'))
            tilenames = sorted(list(set(['_'.join(a.split(os.sep)[-1].split('_')[:-1]) for a in filenames])))
            with open(colFileName, 'w') as file:
                file.write('\n'.join(tilenames))
            
            return tilenames
    
    def getMetadataTiles(self, readcontents = 0):
        # returns path for the metadata of all the tiles.
        # set readcontents = 1 if you want to load the information otherwise returns path
        metDatPath = os.path.join(self.imDirectory, uabCollection.dataDirnames['meta'],uabCollection.metaFilename)
        if(readcontents == 1):
            with open(metDatPath) as f:
                datLines = f.readlines()
                linesByTabs = [line.split('\t') for line in datLines if len(line) > 0]
            return linesByTabs
        else:
            return metDatPath
    
    def setExtensions(self, doSplit=1):
        #function that reads the meta-data file to get all the processed tiles and outputs a dictionary that associates preproc names to extensions
        #if that file doesn't exist, it is made here
        metFileName = self.getMetadataTiles()
        if os.path.exists(metFileName):
            metaContents = self.getMetadataTiles(readcontents=1)
            exts = [a[:2] for a in metaContents]
            self.extensions = {}
            for ext in exts:
                self.extensions[ext[0]] = ext[1]
                self.extDS.append(ext)
        else:
            #this file doesn't exist & the RGB mapping hasn't happened yet so call that here.  If there are additional files to RGB, those remained in Original_Tiles until futher notice and should be specified as such here
            #(1) get all the extensions in the original directory
            allFileTypes = glob.glob(os.path.join(self.imDirectory, uabCollection.dataDirnames['data'], uabCollection.colDirnames['orig'],  self.tileList[0]+'*'))
            exts = list(set([a.split('_')[-1] for a in allFileTypes]))
            
            self.extensions = {}
            for ext in exts:
                self.extensions[ext] = uabCollection.colDirnames['orig']
                self.extDS.append([ext, self.extensions[ext]])
            
            if(doSplit):
                #perform splitting of layers
                kk = list(self.extensions.keys())
                allChans = list(range(len(kk)))
                for chanId in allChans:
                    data = self.loadTileDataByExtension(self.tileList[0], chanId)
                    if(len(data.shape) == 3):
                        #this input is multi-channel so split it.  This also takes care of writing to the meta data file
                        extParts = kk[chanId].split('.')
                        for c in range(data.shape[-1]):
                            extPref = extParts[0] + str(c) + '.' + extParts[1]
                            splitObj = uabPreprocClasses.uabPreprocSplit(chanId, extPref , 'Channel %s Layer %d' % (extParts[0], c), c)
                            splitObj.run(self, forcerun=1)
                    else:
                        extName = kk[chanId]
                        extLocation = self.extensions[extName]
                        self.setMetadataFile("{}\t{}\t{}\n".format(extName, extLocation, 'Original Layer ' + str(chanId)))
            else:
                for cnt, ext in enumerate(exts):
                    self.setMetadataFile("{}\t{}\t{}\n".format(ext, self.extensions[ext], 'Original Layer ' + str(cnt)))   
    
    def readMetadata(self):
        #call this function to get a human readable output of the meta-data relating to the tiles that have been preprocessed
        metFileName = self.getMetadataTiles()
        if os.path.exists(metFileName):
            metaContents = self.getMetadataTiles(readcontents=1)
            print('Description:  these are all the preprocessed tiles available for this dataset.  Use the indexes output on the start of each line to select this tile-type when going to patch extraction in the following step')
            self.extDS = []
            self.extensions = {}
            for cnt, a in enumerate(metaContents):
                self.extensions[a[0]] = a[1]
                self.extDS.append([a[0], a[1]])
                print(('[%d] %s: %s, [ext: %s]' % (cnt, a[2].strip(), a[1].strip(), a[0].strip())))
        else:
            print('This file has not yet been created')
    
    def setMetadataFile(self, updString):
        #update the metadata file
        metFileName = self.getMetadataTiles()
        with open(metFileName,'a+') as file:
            file.write(updString)
            
    def getExtensionInfoById(self, extId):
        #From the meta-data list, get the extension that corresponds to the number extId
        #output: extension, path to extension data
        return self.extDS[extId][0], self.extDS[extId][1]
    
    def loadTileDataByExtension(self, tile, extId):
        #specify extension ID according to the meta data
        ext, dirn = self.getExtensionInfoById(extId)
        tileDataPath = self.getDataNameByTile(dirn, tile, ext)
        return util_functions.uabUtilAllTypeLoad(tileDataPath)
    
    def getDataNameByTile(self, dirn, tileName, ext):
        #convenience function to associate tile name with corresponding data
        return os.path.join(self.imDirectory, uabCollection.dataDirnames['data'], dirn, tileName + '_' + ext)

    def getChannelMeans(self, extId):
        """
        Get means of channels given by extension ids, metainfo has to exist for this function
        :param extId: id of extensions to calculate channel mean, can be a int or list
        :return: np array of meta data
        """
        metaInfoName = os.path.join(self.imDirectory, uabCollection.dataDirnames['meta'], uabCollection.metaInfoFile)
        assert os.path.exists(metaInfoName)
        means = np.zeros(len(extId))

        with open(metaInfoName, 'rb') as f:
            meta = pickle.load(f)
        mean_info = meta['mean']

        for cnt, eid in enumerate(extId):
            means[cnt] = mean_info[eid]

        return means

    def getMetaDataInfo(self, extId, class_info='background,building', forcerun=False):
        """
        Write info to meta.npy, meta data include tile numbers; city list; class num; class info, tile dimension,
        and channel means
        :param extId: id of extensions to calculate channel mean, can be a int or list
        :param class_info: description of classes, split by ','
        :param forcerun: if True, the meta file will be remade
        :return: a dictionary of meta data
        """
        metaInfoName = os.path.join(self.imDirectory, uabCollection.dataDirnames['meta'], uabCollection.metaInfoFile)

        if os.path.exists(metaInfoName) and not forcerun:
            with open(metaInfoName, 'rb') as f:
                meta = pickle.load(f)
        else:
            meta = {}
            # get tile numbers
            tile_names = self.getImLists()
            tile_num = len(tile_names)
            meta['tile_num'] = tile_num

            # get city list
            from string import digits
            remove_digits = str.maketrans('', '', digits)
            city_names = [s.translate(remove_digits) for s in tile_names]
            city_names = list(set(city_names))
            meta['city_names'] = city_names

            # get class num and class info
            class_info = class_info.split(',')
            class_num = len(class_info)
            meta['class_num'] = class_num
            meta['class_info'] = class_info

            # get channel mean
            if type(extId) is not list:
                extId = [extId]
            means = {}
            for ext in extId:
                # load a tile to get shape info
                img = self.loadTileDataByExtension(tile_names[0], ext)
                shape_info = img.shape
                assert len(shape_info) == 2
                channel_mean = 0
                for tile in tqdm(tile_names):
                    img = self.loadTileDataByExtension(tile, ext)
                    channel_mean += np.mean(img)
                means[ext] = channel_mean/tile_num
            meta['mean'] = means

            # save file
            with open(metaInfoName, 'wb') as f:
                pickle.dump(meta, f)
        return meta

    def getAllTileByDirAndExt(self, extId):
        """
        Return a list of tiles as well as the parent directory
        :param extId: id of the extension
        :return: a list of tiles, parent directory of these tiles
        """
        if type(extId) is list:
            dirn = []
            img_list = []
            ext_list = []
            for eid in extId:
                ext, dn = self.getExtensionInfoById(eid)
                img_list.append(self.getImLists(dn))
                dirn.append(os.path.join(self.imDirectory, uabCollection.dataDirnames['data'], dn))
                ext_list.append(ext)
            img_list = [[l[i]+'_'+eid for (l, eid) in zip(img_list, ext_list)] for i in range(len(img_list[0]))]
            if list(set(dirn)) == 1:
                return img_list, list(set(dirn))[0]
            else:
                return img_list, dirn
        else:
            ext, dirn = self.getExtensionInfoById(extId)
            img_list = self.getImLists(dirn)
            return [img+'_'+ext for img in img_list], os.path.join(self.imDirectory, uabCollection.dataDirnames['data'], dirn)
    
"""    
    def getAllTileByDirAndExt(self, dirn, ext):
        imlist = self.getImLists(dirn)
        return [item + self.extensions[ext] for item in imlist]
    
    def getInstanceInfoByTile(self, tileName, mode, toLoad):
        #this loads in the instance level information for the given tile.  This information was generated using external functions which are not being systematized in this class because they are so dataset dependent.  
        #input: tileName, whether to load or just check existence, mode = ('training', testing')
        #output: toLoad = 0 -> info exists? yes-1, no- 0
        #        toLoad = 1 -> info exists? yes-object, no- []
        
        #The output is for a tile is a datastructure with a list of valid pixels at which to extract patches & their label
        
        #make name
        infName = tileName + '_GTlocs.pkl'
        
        fOutput = os.path.join(self.instanceMD, mode, infName)
        code = uabUtilSubm.read_or_new_pickle(fOutput, toLoad, 0)
        
        return code
    
    def computeTrainingMean(self, forceRun=0):
        #computes the mean RGB value of the pixels in training
        
        resDir = os.path.join(self.getResultsDir(), 'collectionInfo','training')
        fname = self.colName+'_mean'
        rgb = uabUtilSubm.read_or_new_pickle(os.path.join(resDir, fname), toLoad=1)
        
        if(not(isinstance(rgb, np.ndarray))):
            #make the directory for the results
            try:
                os.makedirs(resDir)
            except:
                pass
            
            rgb = np.zeros((1,3))
            ind = 0
            for tilename in self.tileTrList:
                cIm = self.loadTileData('training' + uabUtilSubm.sl + tilename)
                cr = cIm.reshape(cIm.shape[0]*cIm.shape[1],cIm.shape[2])
                rgb += np.mean(cr, axis=0)
                ind += 1
                if(np.mod(ind,10) == 0):
                    print 'Finished tile %d' % (ind)
            
            rgb /= len(self.tileTrList)
            
            uabUtilSubm.read_or_new_pickle(os.path.join(resDir, fname), toSave=1,variable_to_save=rgb)
        
        return rgb
    
    def computeClassProportions(self, forceRun=0):
        #computes the class proportions of the pixels in training
        resDir = os.path.join(self.getResultsDir(), 'collectionInfo','training')
        fname = self.colName+'_classProps'
        cprops = uabUtilSubm.read_or_new_pickle(os.path.join(resDir, fname), toLoad=1)
        
        if(not(isinstance(cprops, np.ndarray))):
            #make the directory for the results
            try:
                os.makedirs(resDir)
            except:
                pass
            
            cprops = np.zeros(2)
            ind = 0
            totalPixels = np.prod(self.tileSize)
            for tilename in self.tileTrList:
                cIm = self.loadGTdata('training' + uabUtilSubm.sl + tilename)
                cIm[cIm != 0] = 1
                numTarget = sum(cIm.flatten())
                cprops[0] += (totalPixels - numTarget) / totalPixels
                cprops[1] += numTarget / totalPixels
                
                ind += 1
                if(np.mod(ind,10) == 0):
                    print 'Finished tile %d' % (ind)
            
            cprops /= len(self.tileTrList)
            
            uabUtilSubm.read_or_new_pickle(os.path.join(resDir, fname), toSave=1,variable_to_save=cprops)
            
        return cprops
            
"""     