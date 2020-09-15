#!/bin/bash

python convertData.py $1
mv *.npz /store/user/llavezzo/disappearingTracks/converted_deepSets100_Zee_failAllRecos/
