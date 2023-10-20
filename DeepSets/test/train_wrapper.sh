#!/bin/bash

source /cvmfs/cms.cern.ch/cmsset_default.sh
cd /home/rsantos/scratch0/CMSSW_12_4_11_patch3/src/DisappTrksML/DeepSets/test/

eval `scramv1 runtime -sh`

python3 train.py $1
