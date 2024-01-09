#!/usr/bin/bash

TMPDIR_LOCAL="/data1/$USER/tmp"

if [ ! -d $TMPDIR_LOCAL ]
then 
  mkdir $TMPDIR_LOCAL
fi

export TMPDIR=$TMPDIR_LOCAL
export SINGULARITY_TMPDIR=$TMPDIR
export SINGULARITY_CACHEDIR=$TMPDIR_LOCAL

singularity pull disapp_trks docker://carriganm95/disapp_trks_v2:edited
