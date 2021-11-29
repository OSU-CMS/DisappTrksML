#!/bin/bash

source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=slc7_amd64_gcc820

scramv1 project CMSSW_11_2_1_patch2

cp fakesNN.py CMSSW_11_2_1_patch2/src/fakesNN.py
cp plotMetrics.py CMSSW_11_2_1_patch2/src/plotMetrics.py
cp params.npy CMSSW_11_2_1_patch2/src/params.npy
#cp gridSearchParams.npy CMSSW_11_2_1_patch2/src/gridSearchParams.npy
cp utilities.py CMSSW_11_2_1_patch2/src/utilities.py
cp fakeClass.py CMSSW_11_2_1_patch2/src/fakeClass.py

cd CMSSW_11_2_1_patch2/src/
eval `scramv1 runtime -sh`

#python3 fakesNN.py -d $1 -p gridSearchParams.npy -i $2
if [[$4 -gt 0]]
then
    python3 fakesNN.py -d $1 -p params.npy -i $2 -g $((1/4))
else
    python3 fakesNN.py -d $1 -p params.npy -i $2
fi

echo $((1/4))

rm *.py
rm *.npy

echo $3/$1

if [ -d "$3/$1" ]
then 
    mv * $3/$1/
else
    mkdir $3/$1/
    mv * $3/$1/
fi

cd ..
cd ..

rm -r CMSSW_11_2_1_patch2
