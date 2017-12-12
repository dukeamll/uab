#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 21:03:29 2017

@author: jordan

Example script for defining a collection and running a custom tile-operation that rescales the difference of two tiles
"""

import uab_collectionFunctions
import danielCustom.uabPreprocClasses
import uabPreprocClasses

blCol = uab_collectionFunctions.uabCollection('inria_orgd')

opDetObj = danielCustom.uabPreprocClasses.uabOperTileDiffRescale(13, 7)

rescObj = uabPreprocClasses.uabPreprocMultChanOp([1,2,3], 'RDIFF.tif' , 'Linearly rescale difference between R & B', [1, 2],opDetObj)
rescObj.run(blCol)

blCol.readMetadata()