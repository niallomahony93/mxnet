#!/bin/csh

# specify where the output file should be put
#$ -o /project/dygroup2/xingjian/mxnet

# specify the working path
#$ -wd /project/dygroup2/xingjian/mxnet

#$ -l h=client111

# email me with this address...
#$ -M xshiab@connect.ust.hk
# email when the job starts (b) and after the job has been
# completed (e)
#$ -m be
source ~/.cshrc_user
cd /project/dygroup2/xingjian/opencv-3.2.0/build
#rm -r *
#cmake -D BUILD_opencv_gpu=OFF -D WITH_CUDA=OFF -D WITH_1394=OFF -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/project/dygroup2/xingjian/.local ..
make -j64
