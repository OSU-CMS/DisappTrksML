#!/bin/bash

python3 kfold.py $1
mv gsresults_$1.npy /data/users/llavezzo/cnn/plots/gsresults_$1.npy