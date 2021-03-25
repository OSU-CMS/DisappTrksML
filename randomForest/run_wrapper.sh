#!/bin/bash

source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=slc7_amd64_gcc820

scramv1 project CMSSW_11_2_1_patch2

cp fakesRF.py CMSSW_11_2_1_patch2/src/fakesRF.py

cd CMSSW_11_2_1_patch2/src/
eval `scramv1 runtime -sh`

python3 fakesRF.py

rm *.py
rm *.npy

mkdir /data/users/mcarrigan/$1/
mv * /data/users/mcarrigan/$1/

cd ..
cd ..

rm -r CMSSW_11_2_1_patch2
