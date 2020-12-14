#!/usr/bin/env python

import os
import sys
import math
import ctypes

from ROOT import TChain, TFile, TTree, TH1D, TH2D, TCanvas, THStack, gROOT, TLegend, TGraph, TMarker

from tensorflow.keras.layers import concatenate

from DisappTrksML.DeepSets.architecture import *

def eventSelection( event):
        trackPasses = []
        for track in event.tracks:
            if (abs(track.eta) >= 2.4 or
                track.inGap or
                abs(track.dRMinJet) < 0.5 or
                abs(track.deltaRToClosestElectron) < 0.15 or
                abs(track.deltaRToClosestMuon) < 0.15 or
                abs(track.deltaRToClosestTauHad) < 0.15 or
                (not track.isTagProbeElectron)):
                trackPasses.append(False)
            else:
                trackPasses.append(True)
        return (True in trackPasses), trackPasses

def evaluateFile(inputFileLabel):

	inputFile = TFile(inputFileLabel,"READ")
	tree = inputFile.Get("trackImageProducer/tree")

	nEvents = tree.GetEntries()
	print '\nAdded', nEvents, 'events from file:', inputFileLabel, '\n'

	preds = []

	for iEvent, event in enumerate(tree):
		
		eventPasses, trackPasses = eventSelection(event)

		if not eventPasses: continue

		for iTrack, track in enumerate(event.tracks):
			if not trackPasses[iTrack]: continue
			
			preds.append(arch.evaluate_model(event, track))

	return preds

#######################################

if len(sys.argv) < 4:
	print 'USAGE: python evaluate.py fileIndex fileList inputDir outputDir'
	sys.exit(-1)

fileIndex = sys.argv[1]
fileList = sys.argv[2]
inputDirectory = sys.argv[3]
outputDirectory = sys.argv[4]

inputFileLabel = ""
if int(fileIndex) > 0:
	inarray = np.loadtxt(fileList,dtype=float)
	fileNumber = int(inarray[int(fileIndex)])
	inputFileLabel = "hist_"+str(fileNumber)+".root"

arch = DeepSetsArchitecture()
arch.buildModel()
arch.load_model_weights('model_2020-07-26_13.07.49.h5')

preds, etas = evaluateFile(inputDirectory+inputFileLabel)

np.savez_compressed("preds_"+inputFileLabel+".npz",preds=preds)
os.system('mv -v preds_' + str(inputFileLabel) + '.npz ' + outputDirectory)