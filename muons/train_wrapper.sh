#!/bin/bash

source /cvmfs/cms.cern.ch/cmsset_default.sh
cd /share/scratch0/llavezzo/CMSSW_11_1_3/src/DisappTrksML/muons
eval `scramv1 runtime -sh`

python train.py $1
