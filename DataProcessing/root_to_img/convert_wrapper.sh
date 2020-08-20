#!/bin/bash

python convert_data.py $1
mv *.npz /store/user/mcarrigan/disappearingTracks/images_converted_DYJetsToLL_M50_tanh/
