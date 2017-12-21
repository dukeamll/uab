#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 15:01:14 2017

@author: jordan

Classes defined to split the dataset for training, validation, and testing.
The folds are one number per chip or tile (depending on the input)

The assumption of file names:
    0. Each component (e.g., city name, tile number, patch id, etc) can be concatenated via underscore ('_') or nothing
       if the two components belong to different category (alphabets and digits)
    1. city name comes first, can include alphabetic characters and dash('-')
    2. The tile number follows city name
    3. The patch id follows file names

The file names in fileList can be only the file name of path to the file name, the functions will handle this
"""

import os
import re

def uabUtilGetFolds(parentDir, fileList, xvalType):
    if 'city' == xvalType:
        xvalObj = uabXvalByCity()
    elif 'tile' == xvalType:
        xvalObj = uabXvalByTile()
    elif 'force_tile' == xvalType:
        xvalObj = uabXvalByForceTile()
    else:
        xvalObj = uabXvalParent()
    
    return xvalObj.getFolds(parentDir, fileList)


def concat_list(name_list):
    """
    concatenate a list to string if is a list
    :param name_list: list of names
    :return: a string
    """
    if not isinstance(name_list, str):
        s = ''
        for name in name_list:
            s += name.split('/')[-1]
    else:
        s = name_list.split('/')[-1]
    return s


def getCityName(file_name):
    """
    Get city name of a string, it is assumed that city name is at the beginning of the string and only contain alphas and
    '-'
    :param file_name: file name string, all strings will be concatenated if it is a list
    :return: city name in string
    """
    s = concat_list(file_name)
    return re.findall('^[a-zA-Z\-]*', s)[0]


def getTileNumber(file_name):
    """
    Get tile number of a string, it is assumed that tile number is following the city name and only contains digits
    :param file_name: file name string, all strings will be concatenated if it is a list
    :return: tile name as integer
    """
    s = concat_list(file_name)
    return int(re.findall('[0-9]+', s)[0])


def make_file_list_by_key(idx, file_list, key, filter_list=None):
    if type(key) is not list:
        key = [key]
    if filter_list is None:
        return [file_list[a] for a in range(len(file_list)) if idx[a] in key]
    else:
        if type(filter_list) is not list:
            filter_list = [filter_list]
        file_list_return = []
        for a in range(len(file_list)):
            if idx[a] in key:
                check_flag = 0
                if type(file_list[a]) is list:
                    for item in file_list[a]:
                        for filter_item in filter_list:
                            if filter_item in item:
                                check_flag = 1
                                break
                else:
                    for filter_item in filter_list:
                        if filter_item in file_list[a]:
                            check_flag = 1
                            break
                if check_flag == 0:
                    file_list_return.append(file_list[a])
        return file_list_return
    

class uabXvalParent(object):
    def getFolds(self, parentDir, fileList):
        if(isinstance(fileList, str)):
            filename = os.path.join(parentDir,fileList) 
            with open(filename) as file:
                chipFiles = file.readlines()
            
            chipFiles = [a.strip().split(' ') for a in chipFiles]
        else:
            chipFiles = fileList
        return self.computeFolds(chipFiles), chipFiles
    
    def computeFolds(self, chipFiles):
        #one index per row in the list of lists of chips or tiles by their channel
        raise NotImplementedError('Must be implemented by the subclass')


class uabXvalByCity(uabXvalParent):
    def computeFolds(self, chipFiles):
        idx = []
        cnt = 0
        city_set = {}
        for row in chipFiles:
            city_name = getCityName(row)
            if city_name in city_set:
                # get idx for this city
                idx.append(city_set[city_name])
            else:
                # a new city, update dict and get idx
                city_set[city_name] = cnt
                cnt += 1
                idx.append(city_set[city_name])
        return idx


class uabXvalByTile(uabXvalParent):
    def computeFolds(self, chipFiles):
        idx = []
        cnt = 0
        tile_set = {}
        for row in chipFiles:
            tile_number = getTileNumber(row)
            if tile_number in tile_set:
                # get idx for this city
                idx.append(tile_set[tile_number])
            else:
                # a new city, update dict and get idx
                tile_set[tile_number] = cnt
                cnt += 1
                idx.append(tile_set[tile_number])
        return idx


class uabXvalByForceTile(uabXvalParent):
    """
    Force the idx equal to tile number
    """
    def computeFolds(self, chipFiles):
        idx = []
        for row in chipFiles:
            tile_number = getTileNumber(row)
            idx.append(tile_number)
        return idx
