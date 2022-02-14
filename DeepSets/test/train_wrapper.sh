#!/bin/bash

source /cvmfs/cms.cern.ch/cmsset_default.sh
cd /share/scratch0/mcarrigan/disTracksML/CMSSW_11_2_1_patch2/src/DisappTrksML/DeepSets/test
eval `scramv1 runtime -sh`

python3 train.py $1
