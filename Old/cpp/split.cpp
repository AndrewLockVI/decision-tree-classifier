#include <iostream>

//split_lib = ctypes.CDLL('./cpp/libsplit.so')
//split_lib.split.restype = BestSplit
//split_lib.split.argtypes = [
//        ctypes.POINTER(ctypes.c_float),  # together
//        ctypes.c_float, # prop to train on
//        ctypes.c_float, # prop to val on
//        ctypes.POINTER(ctypes.c_int), # dims
//]
//bestSplit, ltGini, gtGini = split_lib.split(ctypes.POINTER(together), self.propDimsTrain, self.propValSplits, ctypes.POINTER(dims))
//
//
//
//class BestSplit(ctypes.Structure):
//    _fields_ = [("index", ctypes.c_int),
//                ("splitVal", ctypes.c_int),
//                ("ltGini", ctypes.c_float),
//                ("gtGini", ctypes.c_float)
//                ]
//
//


extern "C" {
	struct BestSplit{
		int index;
		float splitVal;
		float ltGini;
		float gtGini;
	};


	BestSplit split(float* together, float propToTrainOn, float propToValWith, int* dims){

		printf("%f\n", together[0]);
		printf("%f\n", propToTrainOn);
		printf("%f\n", propToValWith);
		printf("%i\n", dims[0]);

		return BestSplit();

	}
}
