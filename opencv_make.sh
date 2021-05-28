sudo cmake -D CMAKE_BUILD_TYPE=Release
-D CMAKE_INSTALL_PREFIX=/usr/local
-D BUILD_opencv_python2=OFF
-D BUILD_opencv_python3=ON
-D WITH_CUDA=ON
-D OPENCV_EXTRA_MODULES_PATH=/home/Tom/opencv_contrib-4.5.1/modules
-D PYTHON3_EXECUTABLE=/home/Tom/miniconda3/envs/torch/bin/python
-D CUDA_NVCC_FLAGS=--expt-relaxed-constexpr ..

