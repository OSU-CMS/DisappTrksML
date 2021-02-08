#!/bin/bash

source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=slc7_amd64_gcc820

scramv1 project CMSSW_11_1_2_patch1

cp fakesNN.py CMSSW_11_1_2_patch1/src/fakesNN.py
cp plotMetrics.py CMSSW_11_1_2_patch1/src/plotMetrics.py
cp params.npy CMSSW_11_1_2_patch1/src/params.npy

cd CMSSW_11_1_2_patch1/src/
eval `scramv1 runtime -sh`

python fakesNN.py -d $1 -p params.npy -i $2

rm *.py
rm *.npy

mkdir /data/users/mcarrigan/$1/
mv * /data/users/mcarrigan/$1/

cd ..
cd ..

rm -r CMSSW_11_1_2_patch1
