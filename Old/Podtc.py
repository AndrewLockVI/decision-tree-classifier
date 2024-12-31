from warnings import warn
import ctypes
import numpy as np
from tqdm import tqdm
from SplittingNode import SplittingNode
from SplittingNode import gini
from LeafNode import LeafNode
import math
import graphviz

class PseudoOptimalDecisionTreeClassifier():


    # First is first split, last is ... well yeah.

    def __init__(self, pruneThreshold = .2, maxDepth = 5, proportionToTrainOn=.5, proportionToValidateSplits=.5, proportionOfDimsToTrainOn=.5):
        self.threshold = pruneThreshold 
        self.maxDepth = maxDepth

        # I guess allow > 1, just d

        if(proportionToTrainOn > 1 or proportionToTrainOn <= 0):
            raise Exception(f"Proportion to train on {proportionToTrainOn}, is not valid. Select a proportion in the range of (0,1]")

        self.proportionToTrainOn = proportionToTrainOn

        if(proportionToValidateSplits > 1 or proportionToValidateSplits <= 0):
            raise Exception(f"Proportion to validate splits with {proportionToValidateSplits}, is not valid. Select a proportion in the range of (0,1]")

        self.propValSplits = proportionToValidateSplits

        if(proportionOfDimsToTrainOn > 1 or proportionOfDimsToTrainOn <= 0):
            raise Exception(f"Proportion of dimensions to train on {proportionToValidateSplits}, is not valid. Select a proportion in the range of (0,1]")

        self.propDimsTrain = proportionOfDimsToTrainOn 
        return

    def fit(self, X,  y):

        X,y = self.__validateInput(X,y)
        y_re = y.reshape(-1,1)

        self.sampleSize = X.shape[1]
        self.categories = np.unique(y)
        
        # together [:,-1] == y
        together = np.append(X,y_re, axis=1)

        dims = self.findDims(together)

        self.bestSplit = self.recurse(together, self.maxDepth, dims)

        return

    def findDims(self, together):

        dimCount = len(together[0]) - 1
        dims = np.arange(dimCount)
        dimsToSample = math.ceil(dimCount * self.propDimsTrain)
        if(dimsToSample != dimCount):
            dims = dimsWithMostVar(dimsToSample, together)

        return dims



    def __classification(self, together):
        lastCol = together[:, -1].astype('int')
        counts = np.bincount(lastCol, minlength=len(self.categories))
        majority_label = np.argmax(counts)
        if(len(counts) == 0):
            assert False
        return majority_label, counts



    def recurse(self, together, depth, dims):
        
        if(depth == 0):
            classification, elements = self.__classification(together)
            return LeafNode(classification, len(together), elements)

        bestSplit, ltGini, gtGini = self._best_split(together, self.proportionToTrainOn, dims)

        if bestSplit is None:
            raise ValueError("bestSplit cannot be None")

        ltArr, gtArr = bestSplit.split(arr=together)

        if len(ltArr) == len(together):
            classification, elements = self.__classification(ltArr)
            return LeafNode(classification, len(ltArr), elements)

        if len(gtArr) == len(together):
            classification, elements = self.__classification(gtArr)
            return LeafNode(classification, len(gtArr), elements)

        # might make sense to simply stop
        # if the length of either array is 0
        # because that means splits aren't doing anything..
        # just a thought

        if len(ltArr) > 1 and ltGini > 0:
            blt = self.recurse(ltArr, depth - 1, dims)
            bestSplit.leftChild = blt
        else:
            classification, elements = self.__classification(ltArr)
            bestSplit.leftChild = LeafNode(classification, len(ltArr), elements)

        if len(gtArr) > 1 and gtGini > 0:
            bgt = self.recurse(gtArr, depth - 1, dims)
            bestSplit.rightChild= bgt
        else:
            classification, elements = self.__classification(gtArr)
            bestSplit.rightChild = LeafNode(classification, len(gtArr), elements)

        return bestSplit

    # pass in current root
    # find best options from then on

    # Find best split
    def _best_split(self, together, proportionUsed, dims):

        bestGini = float("inf")
        bestNode  = None
        blg = float("inf")
        bgg = float("inf")

        # indices for evals. This decides which indices to check upon splitting
        values = np.round(np.linspace(0, len(together) - 1, math.ceil(self.propValSplits * len(together)))).astype(np.int32)
        vals = values.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

        indices = np.round(np.linspace(0, len(together) - 2, math.ceil(proportionUsed * len(together)))).astype(int)
        sample_count = len(together[:, 0])

        # columns (excluding y)
        for x in tqdm(dims):

            # random sampling would be a lot faster

            together = together[together[:,x].argsort()]


            # each row (sample)
            # also, we are interpolating between samples

            # indices for splits (this decides how many splits to test)


            eles = together[:, x].astype(np.float32)
            eles = eles.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

            classes = together[:, -1].astype(np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int))

            
            for currentSample in indices:
                splitOn = ((together[currentSample+1][x] - together[currentSample][x]) / 2) + together[currentSample][x]
                split = SplittingNode(x, splitOn)

                current = gini(eles , values, split.val, classes, sample_count, vals)


                if current[0] < bestGini: # type: ignore
                    bestNode = split
                    bestGini = current[0]  # type: ignore
                    blg = current[1]  # type: ignore
                    bgg = current[2]  # type: ignore


        # Return the best node, the left gini impurity, and right gini impurity.
        # These impurities allow for us to stop if we have a pure node.
        return (bestNode, blg, bgg)


    # OPTIMIZE THIS... THIS SHOULD BE DONE WITH BATCHES NOT SINGLE INSTANCES
    def predict(self, X):

        self.__validatePrediction(X)

        y = np.zeros(shape=(len(X),))

        leaf = LeafNode(0,0,[])

        for i in range(0, len(y)):

            done = False
            current = self.bestSplit

            while not done:
                if isinstance(current,type(leaf)):
                    y[i] = current.classification
                    done = True
                    continue

                if self._lessThan(current, X[i]): #type: ignore
                    current = current.leftChild #type: ignore
                else:
                    current = current.rightChild #type: ignore
        return y

    def _lessThan(self, split : SplittingNode, sample):
        if(sample[split.index] < split.val):
            return True
        return False


    def __str__(self):
        return "TODO"

    def __validatePrediction(self, X):

        # check if bestSplit has been set
        try:
            self.bestSplit
        except:
            raise Exception("Tree must be fit prior to prediction")

        X = np.asarray(X)

        if len(X.shape) != 2:
            raise Exception(f"X shape {X.shape} not supported. Ensure input array is 2d.")

        if np.issubdtype(X.dtype, np.str_):
            raise Exception(f"X contains strings which is not allowed.")
        
        if X.shape[1] != self.sampleSize:
            raise Exception(f"Prediction sample of size {X.shape[1]}, not compatible with fitted size of {self.sampleSize}")


        return 0

    def __validateInput(self, X,y):

        y = np.asarray(y)
        X = np.asarray(X)

        if len(X.shape) != 2:
            raise Exception(f"X shape {X.shape} not supported. Ensure input array is 2d.")

        if np.issubdtype(y.dtype, np.str_):
            raise Exception(f"y contains strings which is not allowed.")

        if np.issubdtype(X.dtype, np.str_):
            raise Exception(f"X contains strings which is not allowed.")

        if np.issubdtype(y.dtype, np.floating):  
            if not np.all(np.equal(np.floor(y), y)):  # Check if all values are whole numbers
                raise Exception("y array contains continuous values, but classification requires discrete values")

        if X.shape[0] != y.shape[0]:
            raise Exception(f"Incongruent array sizes. X has shape {X.shape} and y has shape {y.shape}.")
        

        if X.shape[0] <= 1:
            raise Exception(f"X must contain more than one sample.")
        return X,y

    def graph(self):    
        if self.bestSplit == None:
            raise Exception(f"Unable to create graph of classifier, call fit first.")

        graph = graphviz.Digraph()
        graph = createGraph(self.bestSplit, graph)
        graph.render('whatever', format='png', view=True)

def createGraph(node, graph):
    graph.node(str(node.__hash__()), str(node))
    traverseForGraph(node, graph)
    return graph

def traverseForGraph(node, graph):

    cid = str(node.__hash__())

    if node.leftChild == None:
        raise Exception("left child should never be none")

    if node.rightChild == None:
        raise Exception("right child should never be none")

    graph.node(str(node.leftChild.__hash__()), str(node.leftChild))
    graph.node(str(node.rightChild.__hash__()), str(node.rightChild))

    graph.edge(cid, str(node.leftChild.__hash__()))
    graph.edge(cid, str(node.rightChild.__hash__()))


    # will be false in case where child is leaf node
    if type(node.leftChild) == type(node):
        traverseForGraph(node.leftChild, graph)

    if type(node.rightChild) == type(node):
        traverseForGraph(node.rightChild, graph)



    return graph

# use np.var
def dimsWithMostVar(dimCount, arr):
    
    assert dimCount < len(arr[0]) - 1
    
    vars = np.var(arr[:, :-1], axis=0)
    retArr = np.argsort(vars)[::-1]
    retArr = retArr[:dimCount]

    assert dimCount == retArr.shape[0]
    return retArr

