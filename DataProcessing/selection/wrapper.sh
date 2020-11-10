#!/bin/bash

root -l -b -q "makeSelectionReal.cpp+($1,\"$2\",\"$3\")"
mv *.root $4
