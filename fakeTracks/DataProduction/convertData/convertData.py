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
#dataDir = '/store/user/mcarrigan/fakeTracks/selection_v8_aMCNLO/'
#dataDir = "/store/user/mcarrigan/fakeTracks/selection_Z2MuMu_v1/"
dataDir = "/store/user/mcarrigan/fakeTracks/selection_v1_DYJets-MC2022/"
#dataDir = ''

layers = -1

##############
#These pileup cuts are obsolete, matching now done in select script
############

#option to veto if track min(track_vz-pileup_vz) <= pileupCut
pileupVeto = False

#creates pileup_infos for tracks  with vz <= pileupCut
categorical = True

#Threshold for being identified as pileup
pileupCut = 0.1

#############

#Option to include pileup tracks as real tracks
trainPileup = False

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
print("File "+dataDir+fname)

fin = TFile(dataDir+fname, 'read')
#fin = TFile("../selectData/hist_test.root")
fakeTree = fin.Get('fakeTree')
realTree = fin.Get('realTree')
pileupTree = fin.Get('pileupTree')

print("Fake Tree Events: " + str(fakeTree.GetEntries()))
print("Real Tree Events: " + str(realTree.GetEntries()))
print("Pileup Tree Events: " + str(pileupTree.GetEntries()))

def layersEncode(layer, subdet, encodedHits):
    #number of layers in each subdetector (pbx, pex, TIB, TOB, TID, TEC) 
    numLayers = [4, 3, 4, 6, 3, 9]
    #bit = layer-1
    bit = layer
    if(subdet > 1): 
        for sub in range(subdet-1):    
            bit += numLayers[sub]
    print('subdet:', subdet, 'layer', layer, 'bit', bit) 
    encodedHits = encodedHits | 1<<bit
    return encodedHits

def pileupMatching(track):
    min_dz = 10e6
    for vertex in event.pileupZPosition:
        dZ = abs(track.vz - vertex)
        if dZ < min_dz: min_dz = dZ
    print("DZ Matched: ", min_dz)
    return min_dz

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

def signalSelection(track):

    if not (track.trackIso /track.pt < 0.05): return False
    if not (abs(track.d0) < 0.02): return False
    if not (abs(track.dz) < 0.5): return False
    if not (abs(track.dRMinJet) > 0.5): return False

    #candidate track selection
    if not (abs(track.deltaRToClosestElectron) > 0.15): return False
    if not (abs(track.deltaRToClosestMuon) > 0.15): return False
    if not (abs(track.deltaRToClosestTauHad) > 0.15): return False

    #disappearing track selection
    if not (track.missingOuterHits >= 3): return False
    if not (track.ecalo < 10): return False
    return True

def getTrackInfo(track, track_info):
    infos = []
    for info in track_info:
        infos.append(info)
    return infos

def getEventInfo(event, event_info):
    infos = []
    for info in event_info:
        print(info)
        infos.append(info)
    return infos

def getDeDxInfo(hit, hit_info):
    infos = []
    for info in hit_info:
        infos.append(info)
    return infos

def getVertexInfo(vertex, vertex_info):
    infos = []
    for info in vertex_info:
        infos.append(info)
    return infos

def defineEventInfos(network):
    fake_infos = [event.eventNumber, event.nPV]
    if network == 'fakes': return fake_infos
    else: 
        print('Network is not defined in defineEventInfos')
        return 0

def defineTrackInfos(network):
    fake_infos = [track.trackIso, track.eta, track.phi, track.nValidPixelHits, track.nValidHits, track.missingOuterHits, track.dEdxPixel, track.dEdxStrip,                                           track.numMeasurementsPixel, track.numMeasurementsStrip, track.numSatMeasurementsPixel, track.numSatMeasurementsStrip, track.dRMinJet, track.ecalo,                                 track.pt, track.d0, track.dz, track.charge, track.deltaRToClosestElectron, track.deltaRToClosestMuon, track.deltaRToClosestTauHad, track.normalizedChi2]
    if network == 'fakes': return fake_infos
    else:
        print("Network is not defined in defineTrackInfos")
        return 0

def defineDeDxInfos(network):
    fake_infos = [hit.hitLayerId, hit.charge, hit.subDet, hit.pixelHitSize, hit.pixelHitSizeX, hit.pixelHitSizeY, hit.stripShapeSelection, hit.hitPosX, hit.hitPosY]
    if network == 'fakes': return fake_infos
    else:
        print("Network is not defined in defineHitInfos")
        return 0               

def defineVertexInfos(network):
    fake_infos = [vertex.chi2]
    if network == 'fakes': return fake_infos
    else:
        print("Network is not defined in defineVertexInfos")
        return 0

fake_infos, real_infos, pileup_infos = [], [], []
for class_label,tree in zip([0,1,2],[realTree,fakeTree,pileupTree]):

    for iEvent, event in enumerate(tree):
        nPV = event.nPV
        eventNumber = event.eventNumber
        #print(getEventInfo(event, defineEventInfos('fakes')))
        event_info = getEventInfo(event, defineEventInfos('fakes'))

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
           

            track_info = getTrackInfo(track, defineTrackInfos('fakes'))

            passesSelection = signalSelection(track)
            
            track_info = np.concatenate((event_info, [passesSelection], track_info))
            print('track info length', len(track_info))
            
            if(layers == 6 or layers == -1): nLayers = np.zeros((16, 9))
            else: nLayers = np.zeros((layers, 9))
            nLayerCount = 0           
            print("--------------------------------------------------------------------------------------------------------------------------------------------------")
            
            max_energy, min_energy, sum_energy = 0, 10e6, 0
            encodedHits = 0b00000000000000000000000000000

            for iHit, hit in enumerate(track.dEdxInfo):
                if(hit.hitLayerId < 0): continue
                
                layerHits = getDeDxInfo(hit, defineDeDxInfos('fakes'))

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
                    encodedHits = layersEncode(layerHits[0], layerHits[2], encodedHits)
                    nLayerCount += 1

                print('subdet:', layerHits[2], ', layer:', layerHits[0], ', encodedHits:', bin(encodedHits))
 
            for iLayer in range(len(nLayers)):
                print("subDet: " + str(nLayers[iLayer, 2]) + " Layer: " + str(nLayers[iLayer, 0]) + " Charge: " + str(nLayers[iLayer, 1]))

            # Prevent tracks with no hits recorded from continuing
            if(nLayers[0,0] == 0 and nLayers[1,0] == 0): 
                print("not recording track...")
                continue
            
            # Find max layer energy difference and energy sum
            print("sum energy: ", sum_energy, "diff energy:", max_energy-min_energy)
            track_info = np.concatenate((track_info, [sum_energy]))
            track_info = np.concatenate((track_info, [max_energy - min_energy]))
            track_info = np.concatenate((track_info, [encodedHits]))
            
            # add dz and d0 for k nearest neighbors to track
            v_dz, v_d0 = trackMatching(track)
            track_info = np.concatenate((track_info, v_dz[:3]))
            track_info = np.concatenate((track_info, v_d0[:3]))

            #print(track_info)
           
            track_info = np.concatenate((track_info, nLayers.flatten()))
 
            print(len(track_info))

            if(trainPileup):
                if(class_label == 0 or class_label == 2): real_infos.append(track_info)
                if(class_label == 1): fake_infos.append(track_info)

            else:
                if(class_label == 0): real_infos.append(track_info)
                if(class_label == 1): fake_infos.append(track_info)
                if(class_label == 2): pileup_infos.append(track_info)

print("Real Tracks: " + str(len(real_infos)))
print("Fake Tracks: " + str(len(fake_infos)))
print("Pileup Tracks: " + str(len(pileup_infos)))


np.savez_compressed("events_" + str(fileNum) + ".npz", fake_infos = fake_infos, real_infos = real_infos, pileup_infos = pileup_infos)
#np.savez_compressed("events_test.npz", fake_infos = fake_infos, real_infos = real_infos, pileup_infos = pileup_infos)




