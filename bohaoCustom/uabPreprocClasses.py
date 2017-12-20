import numpy as np
import util_functions


# parent class for tile operations.  Has a name and an action
class uabOperTileOps(object):
    def __init__(self, defName):
        self.defaultName = defName

    def getName(self):
        raise NotImplementedError('Must be implemented by the subclass')

    def run(self, tiles):
        raise NotImplementedError('Must be implemented by the subclass')


class uabOperTileDivide(uabOperTileOps):
    def __init__(self, rescFact, defName='Divide'):
        super(uabOperTileDivide, self).__init__(defName)
        self.rescFact = rescFact

    def getName(self):
        return '{}_dF{}'.format(self.defaultName, util_functions.d2s(self.rescFact, 3))

    def run(self, tiles):
        return (tiles[0]/self.rescFact).astype(np.uint8)