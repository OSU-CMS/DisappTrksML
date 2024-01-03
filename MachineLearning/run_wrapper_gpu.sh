#!/usr/bin/bash
localMachine=$(hostname)

echo "Running on computer $localMachine"
echo "Running on GPU"
cp /mnt/driveB/Singularity/disapp_trks.sif .
execute_in_container=$1 # Command that should be executed in container
singularity exec -B $PWD,/store,/data disapp_trks.sif "$execute_in_container"
rm *.sif
