#!/bin/bash

source /cvmfs/cms.cern.ch/cmsset_default.sh
cd /share/scratch0/llavezzo/CMSSW_11_1_3/src/DisappTrksML/DeepSets/test
eval `scramv1 runtime -sh`

python convertToNumpy.py $1 $2 $3 $4
