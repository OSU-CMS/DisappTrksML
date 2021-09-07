#!/bin/bash

python convertDataReal.py $1 $2 $3
mv events_*.npz $4
