#!/usr/bin/bash
localMachine=$(hostname)

# Copy over singularity image 
cp /mnt/driveB/Singularity/disapp_trks.sif .

echo "Running on $localMachine" 

# Execute singularity image
# Change the bind path if you want to run over different data
singularity exec -B /store/user/rsantos/2022/combined_DYJet:/data disapp_trks.sif bash $PWD/ $1 

# Remove singularity image
rm *.sif
