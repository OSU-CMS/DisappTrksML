#!/bin/bash

root -l -b -q "selectData.cpp+($1,\"$2\",\"$3\")"
mv *.root $4
