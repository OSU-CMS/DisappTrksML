#!/bin/bash

python convert_data.py $1 $2
mv *.npz $3
