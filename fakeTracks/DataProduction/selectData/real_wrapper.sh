#!/bin/bash

root -l -b -q "selectDataReal.cpp+($1,\"$2\",\"$3\")"
mv *.root $4
