#!/bin/bash

python make_selection.py $1
mv *.npy /store/user/llavezzo/disappearingTracks/electron_selection_failAllRecos/
mv *.npz /store/user/llavezzo/disappearingTracks/electron_selection_failAllRecos_compressed/

