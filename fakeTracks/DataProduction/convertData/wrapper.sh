#!/bin/bash

python convertData.py $1 $2 $3
mv events_*.npz $4
