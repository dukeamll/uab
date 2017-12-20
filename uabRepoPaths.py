#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 15:22:01 2017

@author: Daniel

This file contains the paths for the results and the data.

Must have a dataPath and resPath defined
"""

from sys import platform
import os

"""
#if you are using the repo on two different computers with different operating systems, this is an example of something you can do
if platform == 'win32':
    #this is the top-level directory with data & results obviously should be changed
    parentDir = 'Y:\\data\\'
elif platform == 'linux2':
    parentDir = '/home/jordan/Daniel/USSOCOM-BDC/dataFiles/'    
"""    

#this example is relevant if you have 1 folder on 1 machine that has both the data and the results.  Your setup may be different
parentDir = r'/media/ei-edl01/data/uab_datasets'
dataPath = parentDir
resPath = r'/hdd/uab_datasets/Results'
modelPath = r'/hdd/Models'
