#!/usr/bin/bash

source /workspace/root_install/bin/thisroot.sh
export PYTHONPATH=$PYTHONPATH:/data/users/mcarrigan/home_link/.local/

if $3; then
    if $4; then
        python3 fakesNN.py -o $1 -c $2 --gpu --scikeras -T
    else
        python3 fakesNN.py -o $1 -c $2 --gpu --scikeras
    fi
else
    echo "Running on cpu"
    source /data/users/mcarrigan/tmp/root/bin/thisroot.sh
    if $4; then
        python3 fakesNN.py -o $1 -c $2 -t --scikeras -T
    else
        python3 fakesNN.py -o $1 -c $2 -t --scikeras
    fi
fi
