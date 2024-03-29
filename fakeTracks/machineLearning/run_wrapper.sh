#!/usr/bin/bash

localMachine=$(hostname)

echo "Running on computer $localMachine"

if $7; then

  if $5; then
    echo "Running on GPU"
    cp /mnt/driveB/Singularity/disapp_trks.sif .
    outDir=$3$1/output_$2
    singularity exec -B $PWD,/store,/data disapp_trks.sif bash $PWD/singularity_wrapper.sh $outDir $6 true $8
    rm *.sif
  
  else
    echo "Running on CPU in singularity"
    outDir=$3$1/output_$2
    singularity exec -B $PWD,/store,/data disappTrksCPU.sif bash $PWD/singularity_wrapper.sh $outDir $6 false $8
    rm *.sif
  fi  
else
  CMSSW_VERSION_LOCAL=CMSSW_12_4_11_patch3

  source /cvmfs/cms.cern.ch/cmsset_default.sh
  export SCRAM_ARCH=slc7_amd64_gcc820

  scramv1 project $CMSSW_VERSION_LOCAL

  cp fakesNN.py $CMSSW_VERSION_LOCAL/src/fakesNN.py
  cp plotMetrics.py $CMSSW_VERSION_LOCAL/src/plotMetrics.py
  cp params.npy $CMSSW_VERSION_LOCAL/src/params.npy
  cp validateData.py $CMSSW_VERSION_LOCAL/src/validateData.py
  #cp gridSearchParams.npy CMSSW_11_2_1_patch2/src/gridSearchParams.npy
  cp utilities.py $CMSSW_VERSION_LOCAL/src/utilities.py
  cp fakeClass.py $CMSSW_VERSION_LOCAL/src/fakeClass.py

  cd $CMSSW_VERSION_LOCAL/src/
  eval `scramv1 runtime -sh`

  #python3 fakesNN.py -d $1 -p gridSearchParams.npy -i $2
  if [[$4 -gt 0]]
  then
    python3 fakesNN.py -d $1 -p params.npy -i $2 -g $((1/4)) -c $6
  else
    python3 fakesNN.py -d $1 -p params.npy -i $2 -c $6
  fi

  echo $((1/4))

  rm *.py
  rm *.npy

  echo $3/$1

  if [ -d "$3/$1" ]; then 
    mv * $3/$1/
  else
    mkdir $3/$1/
    mv * $3/$1/
  fi

  cd ../..

  rm -r $CMSSW_VERSION_LOCAL
fi
