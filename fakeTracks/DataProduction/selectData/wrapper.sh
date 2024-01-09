#!/usr/bin/bash

source ~/root_condor.sh
#CMSSW_VERSION_LOCAL=CMSSW_12_4_11_patch3
#source /cvmfs/cms.cern.ch/cmsset_default.sh
#export SCRAM_ARCH=slc7_amd64_gcc820
#eval `scramv1 runtime -sh`
root -l -b -q "selectData.cpp+($1,\"$2\",\"$3\")"
mv *.root $4
