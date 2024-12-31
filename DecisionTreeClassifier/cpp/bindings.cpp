
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "DecisionTreeClassifier.h"
#include <vector>

namespace py = pybind11;

PYBIND11_MODULE(decision_tree, m) {
    py::class_<DecisionTreeClassifier>(m, "DecisionTreeClassifier")
        .def(py::init<int>())
        .def("fit", [](DecisionTreeClassifier &self, py::array_t<float> X, int samples, py::array_t<int> y, int features) {
            auto X_buf = X.request(); // Request a buffer from NumPy array
            auto y_buf = y.request(); // Request a buffer from NumPy array
            float* X_ptr = static_cast<float*>(X_buf.ptr);
            int* y_ptr = static_cast<int*>(y_buf.ptr);
            self.fit(X_ptr, samples, y_ptr, features);
        })
        .def("predict", [](DecisionTreeClassifier &self, py::array_t<float> X, int samples, int features) {
            auto X_buf = X.request(); // Request a buffer from NumPy array
            float* X_ptr = static_cast<float*>(X_buf.ptr);

            // Get the prediction result as a raw pointer (dynamically allocated array)
            int* result = self.predict(X_ptr, samples, features);


            // Create a NumPy array from the raw pointer
            py::array_t<int> result_array(samples, result);

            // Once the NumPy array is created, delete the dynamically allocated array
            delete[] result;  // Properly deallocate the memory

            return result_array;  // Return the NumPy array
        })
        .def("getDot", &DecisionTreeClassifier::getDot)
        .def("__repr__", [](const DecisionTreeClassifier &dt) {
            return "<DecisionTreeClassifier>";
        });
}

