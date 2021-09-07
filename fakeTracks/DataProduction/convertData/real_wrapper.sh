#!/bin/bash

python convertDataReal.py $1 $2 $3 $5
mv events_*.npz $4
