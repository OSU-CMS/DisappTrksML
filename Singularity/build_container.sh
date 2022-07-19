#!/usr/bin/bash

export TMPDIR=/data1/tmp/
export SINGULARITY_CACHEDIR=/data1/tmp/

singularity build --fakeroot --sandbox osuML build.def
