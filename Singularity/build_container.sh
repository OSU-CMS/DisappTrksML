#!/usr/bin/bash

if [ ! -d "/data1/$USER/tmp" ]
then 
  mkdir /data1/$USER/tmp
fi

export TMPDIR=/data1/$USER/tmp/
export SINGULARITY_CACHEDIR=/data1/$USER/tmp/

singularity build --fakeroot --sandbox osuML build.def
