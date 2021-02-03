#!/usr/bin/env python

import os
import sys
import math
import glob
import pickle
from ROOT import TFile, TTree, gROOT
from ROOT.Math import XYZVector
import numpy as np
import time


fileNum = 1
dataDir = "/store/user/mcarrigan/fakeTracks/selection_v1/"

# script arguments
fileNum = int(sys.argv[1])
if(len(sys.argv)>2): 
    dataDir = str(sys.argv[2])
    if(len(sys.argv)==4):
        fileList = str(sys.argv[3])
        inarray = np.loadtxt(fileList,dtype=float)
        fileNum = int(inarray[fileNum])
fname = "hist_"+str(fileNum)+".root"
print "File "+dataDir+fname 

fin = TFile(dataDir+fname, 'read')
fakeTree = fin.Get('fakeTree')
realTree = fin.Get('realTree')

print(fakeTree.GetEntries())

fake_infos, real_infos = [], []

#fakes are 
for class_label,tree in zip([0,1],[realTree,fakeTree]):

    for iEvent, event in enumerate(tree):
        nPV = event.nPV

        for iTrack, track in enumerate(event.tracks):
            trackIso = track.trackIso
            eta = track.eta
            phi = track.phi
            nValidPixelHits = track.nValidPixelHits
            nValidHits = track.nValidHits
            missingOuterHits = track.missingOuterHits
            #nLayersWithMeasurement = track.nLayersWithMeasurement
            #pixelLayersWithMeasurement = track.pixelLayersWithMeasurement
            dEdxPixel = track.dEdxPixel
            dEdxStrip = track.dEdxStrip
            numMeasurementsPixel = track.numMeasurementsPixel
            numMeasurementsStrip = track.numMeasurementsStrip
            numSatMeasurementsPixel = track.numSatMeasurementsPixel
            numSatMeasurementsStrip = track.numSatMeasurementsStrip
            if(abs(eta) > 1.4): continue
            nLayers = np.zeros((16, 9))
            
            for iHit, hit in enumerate(track.dEdxInfo):
                layerHits = []
                if(hit.hitLayerId == -10): 
                    continue
                layerHits.append(hit.hitLayerId)
                layerHits.append(hit.charge)
                layerHits.append(hit.isPixel)
                layerHits.append(hit.pixelHitSize)
                layerHits.append(hit.pixelHitSizeX)
                layerHits.append(hit.pixelHitSizeY)
                layerHits.append(hit.stripShapeSelection)
                layerHits.append(hit.hitPosX)
                layerHits.append(hit.hitPosY)
            
                #print("Event:", iEvent, "Track:", iTrack, "isFake:", class_label, "hitLayer", hit.hitLayerId, "isPixel:", hit.isPixel, "dEdx:", hit.charge, "Eta:", eta)
                if(nLayers[hit.hitLayerId, 1] == 0 or layerHits[1] > nLayers[hit.hitLayerId, 1]):
                    for i in range(len(layerHits)):
                        nLayers[hit.hitLayerId, i] = layerHits[i]

                


        if(class_label == 0): real_infos.append(nLayers)
        if(class_label == 1): fake_infos.append(nLayers)


np.savez_compressed("events_" + str(fileNum) + ".npz", fake_infos = fake_infos, real_infos = real_infos)




