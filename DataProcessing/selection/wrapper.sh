#!/bin/bash

root -l -b -q 'makeSelection.root('$1','$2','$3')'
mv *.root $3
