#!/bin/bash

source ~/root_condor.sh
source /cvmfs/cms.cern.ch/cmsset_default.sh
eval `scramv1 runtime -sh`

python3 convertData.py $1 $2 $3 $5
mv events_*.npz $4
