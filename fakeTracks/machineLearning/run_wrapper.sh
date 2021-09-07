#!/bin/bash

source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=slc7_amd64_gcc820

scramv1 project CMSSW_11_2_1_patch2

cp fakesNN.py CMSSW_11_2_1_patch2/src/fakesNN.py
cp plotMetrics.py CMSSW_11_2_1_patch2/src/plotMetrics.py
cp params.npy CMSSW_11_2_1_patch2/src/params.npy
cp utilities.py CMSSW_11_2_1_patch2/src/utilities.py

cd CMSSW_11_2_1_patch2/src/
eval `scramv1 runtime -sh`

python3 fakesNN.py -d $1 -p params.npy -i $2

rm *.py
rm *.npy

mkdir /data/users/mcarrigan/fakeTracks/networks/$1/
mv * /data/users/mcarrigan/fakeTracks/networks/$1/

cd ..
cd ..

rm -r CMSSW_11_2_1_patch2
