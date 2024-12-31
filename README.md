# CART Decision Tree Classifier

Simple implementation of CART decision tree classifier written in C++ with hooks into python using pybind11.

Included in this repository (in the 'Old' directory) is another implementation of decision tree classifiers that uses ctypes and varies slightly in implementation. While the current version does not sort input features and then do interpolation, the old implementation does. This could conceivably give better prediction accuracy, but the problem with the implementation is it does not allow for utilization of parallel processing, which the current implementation does. Using the old implementation is unlikely worthwhile given that exhaustive search performed by the CART algorithm, even in performant c++, is quite slow and thus parallellization is likely more important. 
