#!/bin/bash

python convert_data.py $1
mv images_0p25_tanh_$1.npz /store/user/llavezzo/images/images_0p25_tanh_$1.npz