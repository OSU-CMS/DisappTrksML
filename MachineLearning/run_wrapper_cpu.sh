#!/usr/bin/bash
localMachine=$(hostname)
echo "Running on computer $localMachine"
echo "Running on CPU in singularity"
execute_in_container=$1 # Command that should be executed in container
singularity exec -B $PWD,/store,/data disappTrksCPU.sif bash "$execute_in_container"
rm *.sif