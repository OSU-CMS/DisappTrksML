#!/bin/bash

source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=slc7_amd64_gcc820

scramv1 project CMSSW_11_2_1_patch2

cp categoricalNN.py CMSSW_11_2_1_patch2/src/categoricalNN.py
cp plotMetrics.py CMSSW_11_2_1_patch2/src/plotMetrics.py
cp params.npy CMSSW_11_2_1_patch2/src/params.npy

cd CMSSW_11_2_1_patch2/src/
eval `scramv1 runtime -sh`

python3 categoricalNN.py -d $1 -p params.npy -i $2

rm *.py
rm *.npy

mkdir /data/users/mcarrigan/fakeTracks/$1/
mv * /data/users/mcarrigan/fakeTracks/$1/

cd ..
cd ..

rm -r CMSSW_11_2_1_patch2
