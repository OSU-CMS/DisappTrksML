#!/usr/bin/env python

import os
import sys
import glob
import time
from threading import Thread, Lock, Semaphore, active_count
from multiprocessing import cpu_count

from DisappTrksML.DeepSets.architecture import *

## PARAMETERS ##

inputDirectory = '/store/user/bfrancis/images_v5/DYJetsToLL/'

################

arch = DeepSetsArchitecture()

useCondor = False

if len(sys.argv) > 1:
	useCondor = True

if not useCondor:
	print 'Running multiple threads, resulting files stored here...'

	semaphore = Semaphore(cpu_count() + 1)
	printLock = Lock()

	iFile = 0
	threads = []

	for f in glob.glob(inputDirectory + 'hist_*.root'):
		while active_count() > 20:
			time.sleep(1)

		threads.append(Thread(target = arch.convertFileToNumpy, args = (f,)))
		threads[-1].start()

		iFile += 1
		if iFile % 10 == 0:
			printLock.acquire()
			print 'Starting on file:', iFile
			printLock.release()

	for thread in threads:
		thread.join()
else:
	if len(sys.argv) < 4:
		print 'USAGE: python convertToNumpy.py fileNumber inputDir outputDir'
		sys.exit(-1)
		
	fileNumber = sys.argv[1]
	inputDirectory = sys.argv[2]
	outputDirectory = sys.argv[3]

	arch.convertFileToNumpy(inputDirectory + 'hist_' + fileNumber + '.root')
	os.system('mv -v hist_' + fileNumber + '.root.npz ' + outputDirectory)
