#!/bin/bash

python make_selection.py $1
mv *.npy /store/user/mcarrigan/disappearingTracks/electron_selection_V3/
mv *.npz /store/user/mcarrigan/disappearingTracks/electron_selection_V3/

