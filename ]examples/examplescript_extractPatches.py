"""
Created on Thu Dec  7 21:03:29 2017

@author: jordan

Example script for extracting patches from a collection.

The numbers in this file relate to a particular type of U-net
"""

import uabRepoCode.uab_collectionFunctions
import uabRepoCode.uab_DataHandlerFunctions

blCol = uabRepoCode.uab_collectionFunctions.uabCollection('inria_orgd')
extrObj = uabRepoCode.uab_DataHandlerFunctions.uabPatchExtr([0,1,2], cSize=(572,572), numPixOverlap=92,extSave=['png','jpg','jpg'])

extrObj.run(blCol)