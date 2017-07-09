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
make -j64
