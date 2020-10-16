#!/bin/bash

source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=slc7_amd64_gcc820

scramv1 project CMSSW_11_1_2_patch1

cp train.py CMSSW_11_1_2_patch1/src/train.py
cp utils.py CMSSW_11_1_2_patch1/src/utils.py
cp validate.py CMSSW_11_1_2_patch1/src/validate.py
cp params.npy CMSSW_11_1_2_patch1/src/params.npy
cp generator.py CMSSW_11_1_2_patch1/src/generator.py
cp model.py CMSSW_11_1_2_patch1/src/model.py

cd CMSSW_11_1_2_patch1/src/
eval `scramv1 runtime -sh`

python3 train.py -d $1 -p params.npy -i $2 >> log.txt

rm *.py
rm *.npy

mkdir /data/users/llavezzo/$1/
mv $1 /data/users/llavezzo/$1/

cd ..
cd ..

rm -r CMSSW_11_1_2_patch1