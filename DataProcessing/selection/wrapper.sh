#!/bin/bash

root -l -b -q "makeSelection.cpp+($1,\"$2\",\"$3\")"
mv *.root $4
