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

layers = 4

# script arguments
if(len(sys.argv)>1): fileNum = int(sys.argv[1])
if(len(sys.argv)>2): 
    dataDir = str(sys.argv[2])
    if(len(sys.argv)>3):
        fileList = str(sys.argv[3])
        inarray = np.loadtxt(fileList,dtype=float)
        fileNum = int(inarray[fileNum])
        layers = int(sys.argv[4])
fname = "hist_"+str(fileNum)+".root"
print "File "+dataDir+fname 

fin = TFile(dataDir+fname, 'read')
fakeTree = fin.Get('fakeTree')
realTree = fin.Get('realTree')

print(fakeTree.GetEntries())

fake_infos, real_infos = [], []
fake_d0, real_d0 = [], []
#fakes are 
for class_label,tree in zip([0,1],[realTree,fakeTree]):

    for iEvent, event in enumerate(tree):
        nPV = event.nPV

        for iTrack, track in enumerate(event.tracks):
            nLayersWithMeasurement = track.nLayersWithMeasurement
            if(nLayersWithMeasurement == layers): continue
            eta = track.eta
            if(abs(eta) > 1.4): continue
            trackd0 = track.d0
            if(class_label==1): fake_d0.append(trackd0)
            if(class_label==0): real_d0.append(trackd0)
            trackIso = track.trackIso
            phi = track.phi
            nValidPixelHits = track.nValidPixelHits
            nValidHits = track.nValidHits
            missingOuterHits = track.missingOuterHits
            dEdxPixel = track.dEdxPixel
            dEdxStrip = track.dEdxStrip
            numMeasurementsPixel = track.numMeasurementsPixel
            numMeasurementsStrip = track.numMeasurementsStrip
            numSatMeasurementsPixel = track.numSatMeasurementsPixel
            numSatMeasurementsStrip = track.numSatMeasurementsStrip
            dRMinJet = track.dRMinJet
            ecalo = track.ecalo
            pt = track.pt
            d0 = track.d0
            dz = track.dz
            charge = track.charge
            
            track_info = [trackIso, eta, phi, nPV, dRMinJet, ecalo, pt, d0, dz, charge, nValidPixelHits, nValidHits, missingOuterHits, dEdxPixel, dEdxStrip, numMeasurementsPixel, 
                          numMeasurementsStrip, numSatMeasurementsPixel, numSatMeasurementsStrip]
            
            if(layers == 6): nLayers = np.zeros((16, 9))
            else: nLayers = np.zeros((layers, 9))
            
            for iHit, hit in enumerate(track.dEdxInfo):
                layerHits = []
                if(hit.hitLayerId < 0 or hit.hitLayerId >= layers): 
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
                
                print("Event: " + str(iEvent) + ", Track: " + str(iTrack) + ", Layer: " + str(hit.hitLayerId))
                if(nLayers[hit.hitLayerId, 1] == 0 or layerHits[1] > nLayers[hit.hitLayerId, 1]):
                    for i in range(len(layerHits)):
                        nLayers[hit.hitLayerId, i] = layerHits[i]

            #require that layers 0-x are filled
            layers_filled = True
            for layer in range(layers):
                if(layers == 6 and layer > 5): break
                if(nLayers[layer, 0]==0 and nLayers[layer, 1]==0): layers_filled = False
            
            if(layers_filled == False): continue

            track_info = np.concatenate((track_info, nLayers.flatten()))

            if(class_label == 0): real_infos.append(track_info)
            if(class_label == 1): fake_infos.append(track_info)


print("Real Tracks: " + str(len(real_infos)))
print("Fake Tracks: " + str(len(fake_infos)))

np.savez_compressed("events_" + str(fileNum) + ".npz", fake_infos = fake_infos, real_infos = real_infos, fake_d0 = fake_d0, real_d0 = real_d0)




