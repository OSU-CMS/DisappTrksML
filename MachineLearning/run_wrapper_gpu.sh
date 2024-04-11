#!/usr/bin/bash
localMachine=$(hostname)

echo "Running on computer $localMachine"
echo "Running on GPU"
cp /mnt/driveB/Singularity/disapp_trks.sif .
execute_in_container=$@ # Command that should be executed in container
echo "Running $execute_in_container"
singularity exec -B $PWD,/store,/data disapp_trks.sif bash $PWD/singularity_wrapper.sh $1 
rm *.sif
