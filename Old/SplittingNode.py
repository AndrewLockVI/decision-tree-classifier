import numpy as np
import math
import ctypes

class SplittingNode:

    def __init__(self, feature_index, value, rightChild = None, leftChild = None):
        self.index = feature_index
        self.val = value

        self.rightChild = rightChild
        self.leftChild = leftChild

    # maybe add input validation???
    # do in place weighted gini calculation

    # split the data by current node


    def split(self, arr):

        pySplit = True

        if pySplit:
            return self.__split_py(arr) 
        else:
            assert False



    def __split_py(self, arr):
        ltCount = 0
        gtCount = 0

        for i in range(0, len(arr)):
            lessThan = _lessThan(arr[i], self.index, self.val)
            if lessThan:
                ltCount += 1
            else:
                gtCount += 1



        ltArr = np.zeros(shape=(ltCount, arr.shape[1]))
        gtArr = np.zeros(shape=(gtCount, arr.shape[1]))

        ltItr = 0
        gtItr = 0

        for i in range(0, len(arr)):
            lt = _lessThan(arr[i], self.index, self.val)
            if lt:
                ltArr[ltItr] = arr[i]
                ltItr += 1
            else:
                gtArr[gtItr] = arr[i]
                gtItr += 1
        
        assert ltItr + gtItr == len(arr)

        return ltArr, gtArr


    def __str__(self):
        return f"Splitting index: {self.index}\nSplitting value: {round(self.val,2)}"

class GiniResult(ctypes.Structure):
    _fields_ = [("weighted", ctypes.c_float),
                ("ltGini", ctypes.c_float),
                ("gtGini", ctypes.c_float)]

def gini(eles, values, val, classes, sample_count, vals):

    gini_lib = ctypes.CDLL('./cpp/libgini.so')
    gini_lib.gini.restype = GiniResult
    gini_lib.gini.argtypes = [
            ctypes.POINTER(ctypes.c_float), 
            ctypes.POINTER(ctypes.c_int), 
            ctypes.c_int, 
            ctypes.c_float, 
            ctypes.POINTER(ctypes.c_int), 
            ctypes.c_int
    ]

    split_val = ctypes.c_float(val)
    result = gini_lib.gini(eles, classes, sample_count, split_val, vals, len(values))
    weightedGini = result.weighted
    ltGini = result.ltGini
    gtGini = result.gtGini
    return (weightedGini, ltGini, gtGini)


def giniPy(combined , values, index, val):
    ltc = {}
    geqc = {}
    ltCount = 0
    geqCount = 0
    for i in values:

        lt = _lessThan(combined[i], index, val)
        classification = int(combined[i][-1])

        if(lt):
            ltCount += 1
            value = ltc.get(classification)
            if(value != None):
                ltc[classification] = value + 1
            else:
                ltc[classification] = 1
        else:
            geqCount += 1
            value = geqc.get(classification)
            if(value != None):
                geqc[classification] = value + 1
            else:
                geqc[classification] = 1

    lt_gini = 1
    for key in ltc.keys():
        lt_gini -= (ltc[key] / ltCount)**2

    gt_gini = 1
    for key in geqc.keys():
        gt_gini -= (geqc[key] / geqCount)**2

    if(geqCount == 0):
        gt_gini = 0
    if(ltCount == 0):
        lt_gini = 0

    lt_percent = ltCount / len(values)
    gt_percent = geqCount / len(values)

    weighted_gini = (lt_gini * lt_percent) + (gt_gini * gt_percent)

    return (weighted_gini, lt_gini, gt_gini)


def _lessThan(sample, index, val):
    value = sample[index]
    if(value < val):
        return True
    return False

