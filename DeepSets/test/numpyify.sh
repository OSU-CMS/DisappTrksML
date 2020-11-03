#!/bin/bash

source /cvmfs/cms.cern.ch/cmsset_default.sh
cd /share/scratch0/bfrancis/disappTrks/CMSSW_9_4_9/src/DisappTrksML/DeepSets/test/
eval `scramv1 runtime -sh`

python convertToNumpy.py $1 $2 $3
