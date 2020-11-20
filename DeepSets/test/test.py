#!/usr/bin/env python

import os
import sys

from DisappTrksML.DeepSets.architecture import *

f = '/store/user/bfrancis/images_v5/DYJetsToLL/hist_929.root'

arch = DeepSetsArchitecture()

arch.convertFileToNumpy(f)
