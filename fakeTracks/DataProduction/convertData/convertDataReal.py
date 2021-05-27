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
#dataDir = "/store/user/mcarrigan/fakeTracks/selection_Z2MuMu_v1/"
#dataDir = '/store/user/mcarrigan/fakeTracks/selection_ZeroBias_v8/'
dataDir = '../selectData/'

layers = -1

#option to veto if track min(track_vz-pileup_vz) <= pileupCut
pileupVeto = False

#creates pileup_infos for tracks  with vz <= pileupCut
categorical = True

#Threshold for being identified as pileup
pileupCut = 0.1

# script arguments
if(len(sys.argv)>1):
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
realTree = fin.Get('realTree')

print(realTree.GetEntries())

def trackMatching(track):
    v_dz = [10e6, 10e6, 10e6, 10e6, 10e6]
    v_d0 = [10e6, 10e6, 10e6, 10e6, 10e6]
    for v in event.vertexInfos:
        #print(v.vertex.Z())
        dZ = abs(track.vz - v.vertex.Z())
        d0 = np.sqrt((track.vx - v.vertex.X())**2 + (track.vy - v.vertex.Y())**2)
        v_dz.append(dZ)
        v_d0.append(d0)
    v_dz.sort()
    v_d0.sort()
    #print(v_d0[0], track.d0)
    return v_dz, v_d0

def pileupMatching(track):
    min_dz = 10e6
    for vertex in event.pileupZPosition:
        dZ = abs(track.vz - vertex)
        if dZ < min_dz: min_dz = dZ
    print("DZ Matched: ", min_dz)
    return min_dz

real_infos = []
d0 = []

#fakes are 
for class_label,tree in zip([0],[realTree]):

    for iEvent, event in enumerate(tree):
        nPV = event.nPV

        for iTrack, track in enumerate(event.tracks):
  
            nLayersWithMeasurement = track.nLayersWithMeasurement
            #case for 4, 5 layers to pass
            if(layers < 6 and layers >= 4):
                if(nLayersWithMeasurement != layers): continue
            #case to allow >=6 to pass
            elif(layers == 6):
                if(nLayersWithMeasurement < layers): continue
            #case to allow all >=4 to pass
            elif(layers == -1):
                if(nLayersWithMeasurement < 4): continue
            eta = track.eta
            if(abs(eta) > 2.4): continue
  
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

            track_info = [trackIso, eta, phi, nPV, dRMinJet, ecalo, pt, d0, dz, charge, nValidPixelHits, nValidHits, missingOuterHits, dEdxPixel, dEdxStrip, 
                          numMeasurementsPixel, numMeasurementsStrip, numSatMeasurementsPixel, numSatMeasurementsStrip]

            if(layers == 6 or layers == -1): nLayers = np.zeros((16, 9))
            else: nLayers = np.zeros((layers, 9))
            nLayerCount = 0
            print("--------------------------------------------------------------------------------------------------------------------------------------------------")

            max_energy, min_energy, sum_energy = 0, 10e6, 0

            for iHit, hit in enumerate(track.dEdxInfo):
                layerHits = []
                print("Event: " + str(iEvent) + ", Track: " + str(iTrack) + " isFake: " + str(class_label) + ", subDet: " + str(hit.subDet) + " Layer: "
                        + str(hit.hitLayerId), "nLayersWithMeasurement: " + str(nLayersWithMeasurement) + " Eta: " + str(track.eta) + " missingInnerHits: "
                        + str(track.missingInnerHits) + " missingMiddleHits: " + str(track.missingMiddleHits) + " missingOuterHits: " + str(track.missingOuterHits) +
                        " Charge: " + str(hit.charge))
                if(hit.hitLayerId < 0):
                    continue
                layerHits.append(hit.hitLayerId)
                layerHits.append(hit.charge)
                layerHits.append(hit.subDet)
                layerHits.append(hit.pixelHitSize)
                layerHits.append(hit.pixelHitSizeX)
                layerHits.append(hit.pixelHitSizeY)
                layerHits.append(hit.stripShapeSelection)
                layerHits.append(hit.hitPosX)
                layerHits.append(hit.hitPosY)

                if(hit.charge > max_energy): max_energy = hit.charge
                if(hit.charge < min_energy): min_energy = hit.charge
                sum_energy += hit.charge

                newLayer = True
                if(layers == 6 or layers == -1):
                    for iSaved, savedHit in enumerate(nLayers):
                        if(hit.subDet == savedHit[2] and hit.hitLayerId == savedHit[0]):
                            newLayer = False
                            if (hit.charge > savedHit[1]):
                                for i in range(len(layerHits)):
                                    nLayers[iSaved, i] = layerHits[i]

                if(newLayer==True):
                    if(nLayerCount > len(nLayers)-1): continue
                    for i in range(len(layerHits)):
                        nLayers[nLayerCount, i] = layerHits[i]
                    nLayerCount += 1
            for iLayer in range(len(nLayers)):
                print("subDet: " + str(nLayers[iLayer, 2]) + " Layer: " + str(nLayers[iLayer, 0]) + " Charge: " + str(nLayers[iLayer, 1]))

            # Prevent tracks with no hits recorded from continuing
            if(nLayers[0,0] == 0 and nLayers[1,0] == 0):
                print("not recording track...")
                continue

            # Find max layer energy difference and energy sum
            print("sum energy: ", sum_energy, "diff energy:", max_energy-min_energy)
            track_info.append(sum_energy)
            track_info.append(max_energy - min_energy)

            # add dz and d0 for k nearest neighbors to track
            v_dz, v_d0 = trackMatching(track)
            track_info = np.concatenate((track_info, v_dz[:3]))
            track_info = np.concatenate((track_info, v_d0[:3]))

            #print(track_info)

            track_info = np.concatenate((track_info, nLayers.flatten()))

            if(class_label == 0): real_infos.append(track_info)


print("Real Tracks: " + str(len(real_infos)))

#for i in range(len(real_infos)):
#    print(real_infos[i])
#    print("#####################################################################################################################################")

np.savez_compressed("events_" + str(fileNum) + ".npz", real_infos = real_infos, pileup_infos = pileup_infos)



