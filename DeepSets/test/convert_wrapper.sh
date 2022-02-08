#!/bin/bash

source /cvmfs/cms.cern.ch/cmsset_default.sh
#cd /share/scratch0/llavezzo/CMSSW_11_1_3/src/DisappTrksML/DeepSets/test
cd /share/scratch0/mcarrigan/disTracksML/CMSSW_11_2_1_patch2/src/DisappTrksML/DeepSets/test
eval `scramv1 runtime -sh`

python3 convertToNumpy.py $1 $2 $3 $4
