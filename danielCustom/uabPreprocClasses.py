#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 13:08:49 2017

@author: jordan
"""
import util_functions        

# parent class for tile operations.  Has a name and an action
class uabOperTileOps(object):
    def __init__(self, defName):
        self.defaultName = defName
        
    def getName(self):
        raise NotImplementedError('Must be implemented by the subclass')
    
    def run(self, tiles):
        raise NotImplementedError('Must be implemented by the subclass')

class uabOperTileDiffRescale(uabOperTileOps):
    def __init__(self, rescFact, rescBias, defName = 'DiffResc'):
        super(uabOperTileDiffRescale, self).__init__(defName)
        self.rescFact = rescFact
        self.rescBias = rescBias
    
    def getName(self):
        return '%s_rF%s_rB%s' % (self.defaultName, util_functions.d2s(self.rescFact,3), util_functions.d2s(self.rescBias,3))
    
    def run(self, tiles):
        return self.rescFact * (tiles[1] - tiles[0]) + self.rescBias