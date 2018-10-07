# Locates the tensorFlow library and include directories.

include(FindPackageHandleStandardArgs)

# For tensorflow 1.6 (include)
list(APPEND TensorFlow_INCLUDE_DIR /home/bryan/tensorflow/local/include/google/tensorflow)
list(APPEND TensorFlow_INCLUDE_DIR /home/bryan/tensorflow/bazel-tensorflow/external/nsync/public)

# For tensorflow 1.6 (libs)
set(TensorFlow_LIBRARIES /home/bryan/tensorflow/local/lib/libtensorflow_all.so)

# set TensorFlow_FOUND
find_package_handle_standard_args(TensorFlow DEFAULT_MSG TensorFlow_INCLUDE_DIR TensorFlow_LIBRARIES)

# set external variables for usage in CMakeLists.txt
if(TENSORFLOW_FOUND)
    set(TensorFlow_LIBRARIES ${TensorFlow_LIBRARIES})
    set(TensorFlow_INCLUDE_DIRS ${TensorFlow_INCLUDE_DIR})
    message(STATUS "TensorFlow found (include: ${TensorFlow_INCLUDE_DIRS})")
    message(STATUS "TensorFlow found (lib: ${TensorFlow_LIBRARIES})")
endif()
