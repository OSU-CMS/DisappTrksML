#!/bin/bash

source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=slc6_amd64_gcc630

scramv1 project CMSSW CMSSW_10_2_20

cp kfold.py CMSSW_10_2_20/src/kfold.py
cp utils.py CMSSW_10_2_20/src/utils.py
cp cnn.py CMSSW_10_2_20/src/cnn.py

cd CMSSW_10_2_20/src/
eval `scramv1 runtime -sh`

python3 kfold.py $1
mv gsresults_$1.npy /data/users/llavezzo/cnn/plots/gsresults_$1.npy

cd ..
cd ..
rm -r CMSSW_10_2_20