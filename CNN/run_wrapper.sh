#!/bin/bash

source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=slc7_amd64_gcc820

scramv1 project CMSSW_11_1_2_patch1

cp flow.py CMSSW_11_1_2_patch1/src/flow.py
cp utils.py CMSSW_11_1_2_patch1/src/utils.py
cp validate.py CMSSW_11_1_2_patch1/src/validate.py

cd CMSSW_11_1_2_patch1/src/
eval `scramv1 runtime -sh`

python3 flow.py $1
mv $1 /data/users/llavezzo/cnn/

cd ..
cd ..
rm -r CMSSW_11_1_2_patch1