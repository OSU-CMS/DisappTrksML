#!/usr/bin/env python

import os
import sys
import glob
import time
from threading import Thread, Lock, Semaphore, active_count
from multiprocessing import cpu_count

from DisappTrksML.SiameseNetwork.SiameseNetwork import *

## PARAMETERS ##

inputDirectory = '/store/user/mcarrigan/Images-v7p1-DYJets-MC2017_aMCNLO/'
outputDirectory = ""
fileNumber = 260

################

arch = SiameseNetwork(eta_range=1.0,phi_range=1.0,max_hits=20)

useCondor = False
useMultiThreads = False

if len(sys.argv) > 1:
	useCondor = True

if useMultiThreads:
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

elif useCondor:
	if len(sys.argv) < 4:
		print 'USAGE: python convertToNumpy.py fileIndex fileList inputDir outputDir'
		sys.exit(-1)
		
	fileIndex = sys.argv[1]
	fileList = sys.argv[2]
	inputDirectory = sys.argv[3]
	outputDirectory = sys.argv[4]

	inarray = np.loadtxt(fileList,dtype=float)
	fileNumber = int(inarray[int(fileIndex)])

	arch.convertAMSBFileToNumpy(inputDirectory + 'hist_' + str(fileNumber) + '.root')
	os.system('mv -v hist_' + str(fileNumber) + '.root.npz ' + outputDirectory)

else:
	arch.convertAMSBFileToNumpy(inputDirectory + 'hist_' + str(fileNumber) + '.root')
	os.system('mv -v hist_' + str(fileNumber) + '.root.npz ' + outputDirectory)