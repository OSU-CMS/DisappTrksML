#!/bin/bash
# cd /home/rsantos/scratch0/CMSSW_12_4_11_patch3/src/DisappTrksML/DeepSets/test
# cp /mnt/driveB/Singularity/disapp_trks.sif .
singularity exec -B /store,/data disapp_trks.sif bash test/train_singularity_wrapper.sh 0
#singularity shell -B /home/rsantos/scratch0/CMSSW_12_4_11_patch3/src/DisappTrksML:/home/ryan/scratch0/CMSSW_12_4_11_patch3/src/DisappTrksML,/store,/data disapp_trks.sif 
