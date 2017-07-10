#!/bin/csh

# specify where the output file should be put
#$ -o /project/dygroup2/xingjian/mxnet
#$ -e /project/dygroup2/xingjian/mxnet

# specify the working path
#$ -wd /project/dygroup2/xingjian/opencv-3.2.0/build

#$ -l h=client112

# email me with this address...
#$ -M xshiab@connect.ust.hk
# email when the job starts (b) and after the job has been
# completed (e)
#$ -m be
source ~/.cshrc_user
cd /project/dygroup2/xingjian/opencv-3.2.0/build
rm -rf *
cmake -D CMAKE_BUILD_TYPE=RELEASE -D BUILD_ZLIB=ON -D BUILD_OPENEXR=ON -D BUILD_ILMIMF=ON -D BUILD_TBB=ON -D BUILD_JASPER=ON -D BUILD_PNG=ON -D BUILD_JPEG=ON -D BUILD_TIFF=ON -D BUILD_opencv_gpu=OFF -D WITH_CUDA=OFF -D WITH_1394=OFF -DBUILD_opencv_java=OFF -D BUILD_OPENCV_PYTHON2=OFF -D BUILD_OPENCV_PYTHON3=OFF -D MKL_ROOT_DIR=/project/dygroup2/xingjian/intel/mkl -D CMAKE_INSTALL_PREFIX=/project/dygroup2/xingjian/.local ..
make -j64
