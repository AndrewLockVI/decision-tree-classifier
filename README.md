# CART Decision Tree Classifier

Simple implementation of CART decision tree classifier written in C++ with hooks into python using pybind11.

Included in this repository (in the 'Old' directory) is another implementation of a decision tree classifier that uses ctypes and varies slightly in implementation. While the current version does not sort input features to do interpolation to optimize splitting, the old implementation does. This would give the old implementation nominally better prediction accuracy, but the problem with that implementation is it does not allow for utilization of parallel processing, which the current implementation does. Using the old implementation is unlikely worthwhile given that exhaustive search performed by the CART algorithm, even in performant c++, is quite slow and thus parallellization is likely more valuable. 
