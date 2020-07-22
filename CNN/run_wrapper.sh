#!/bin/bash

source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=slc6_amd64_gcc630

scramv1 project CMSSW CMSSW_10_2_20

cp flow.py CMSSW_10_2_20/src/flow.py
cp utils.py CMSSW_10_2_20/src/utils.py
cp validate.py CMSSW_10_2_20/src/validate.py

cd CMSSW_10_2_20/src/
eval `scramv1 runtime -sh`

python3 flow.py $1
mv $1 /data/users/llavezzo/cnn/

cd ..
cd ..
rm -r CMSSW_10_2_20