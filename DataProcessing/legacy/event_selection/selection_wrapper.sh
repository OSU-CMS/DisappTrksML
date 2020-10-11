#!/bin/bash

python make_selection.py $1 $2
mv *.npy $3
mv *.npz $3

