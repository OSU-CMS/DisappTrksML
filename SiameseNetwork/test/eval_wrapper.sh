#!/bin/bash

source /cvmfs/cms.cern.ch/cmsset_default.sh
cd /share/scratch0/llavezzo/CMSSW_11_1_3/src/DisappTrksML/DeepSets/test
eval `scramv1 runtime -sh`

python evaluate.py $1 $2 $3 $4 $5
