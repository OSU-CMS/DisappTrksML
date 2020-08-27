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
from threading import Thread, Lock, Semaphore, active_count
from multiprocessing import cpu_count

gROOT.ProcessLine('.L ' + os.environ['CMSSW_BASE'] + '/src/DisappTrksML/TreeMaker/interface/Infos.h+')

eta_range = 0.5
phi_range = 0.5

dataMode = True # if true, use tagProbe selection; if false, use genmatched pdgID

# combine EB+EE and muon detectors into ECAL/HCAL/MUO indices
def detectorIndex(detType):
    if detType == 1 or detType == 2:
        return 0
    elif detType == 4:
        return 1
    elif detType >= 5 and detType <= 7:
        return 2
    else:
        return -1

# track is close to a reconstructed lepton of flavor
def isReconstructed(track, flavor):
    if flavor is 'ele':
        return abs(track.deltaRToClosestElectron) < 0.15
    elif flavor is 'muon':
        return abs(track.deltaRToClosestMuon) < 0.15
    elif flavor is 'tau':
        return abs(track.deltaRToClosestTauHad) < 0.15
    else:
        return False

# track is gen-matched to pdgId
def isGenMatched(track, pdgId):
    return (abs(track.genMatchedID) == pdgId and abs(track.genMatchedDR) < 0.1)

# return (dEta, dPhi) between track and hit
def imageCoordinates(track, hit):
    momentum = XYZVector(track.px, track.py, track.pz)
    track_eta = momentum.Eta()
    track_phi = momentum.Phi()
    dEta = track_eta - hit.eta
    dPhi = track_phi - hit.phi
    # branch cut [-pi, pi)
    if abs(dPhi) > math.pi:
        dPhi -= round(dPhi / (2. * math.pi)) * 2. * math.pi
    return (dEta, dPhi)

threads = []
printLock = Lock()
semaphore = Semaphore(cpu_count() + 1)

eventsProcessed = 0

maxHitsInImages = 1000

# process fileName, appending to images and labels
def processFile(fileName):
    global printLock
    global semaphore
    global eventsProcessed

    semaphore.acquire()

    trackData = [] # [(image, label)]
    # where image: [(dEta, dPhi, energy, detIndex)]
    # label=1: electron, label=0: not electron

    fin = TFile(fileName, 'read')
    tree = fin.Get('trackImageProducer/tree')

    nReconstructed = tree.Draw('tracks.px', 'abs(tracks.deltaRToClosestElectron) < 0.15', 'goff')
    nNotReconstructed = tree.Draw('tracks.px', 'abs(tracks.deltaRToClosestElectron) >= 0.15', 'goff')

    images_reco = np.zeros(shape=(nReconstructed, maxHitsInImages, 4))
    images_fail = np.zeros(shape=(nNotReconstructed, maxHitsInImages, 4))

    labels_reco = np.zeros(shape=(nReconstructed,))
    labels_fail = np.zeros(shape=(nNotReconstructed,))

    iReco = 0
    iFail = 0

    for iTrack, event in enumerate(tree):
        for track in event.tracks:
            imageHits = []
            for hit in event.recHits:
                dEta, dPhi = imageCoordinates(track, hit)
                if abs(dEta) < eta_range and abs(dPhi) < phi_range:
                    detIndex = detectorIndex(hit.detType)
                    if detIndex < 0:
                        continue
                    energy = hit.energy if detIndex != 2 else 1
                    imageHits.append((dEta, dPhi, energy, detIndex))
            
            if isReconstructed(track, 'ele'):
                labels_reco[iReco] = int(track.isTagProbeElectron) if dataMode else int(isGenMatched(track, 11))
                for iHit in range(min(len(imageHits), maxHitsInImages)):
                    images_reco[iReco][iHit][0] = imageHits[iHit][0]
                    images_reco[iReco][iHit][1] = imageHits[iHit][1]
                    images_reco[iReco][iHit][2] = imageHits[iHit][2]
                    images_reco[iReco][iHit][3] = imageHits[iHit][3]
                iReco += 1
            else:
                labels_fail[iFail] = int(track.isTagProbeElectron) if dataMode else int(isGenMatched(track, 11))
                for iHit in range(min(len(imageHits), maxHitsInImages)):
                    images_fail[iFail][iHit][0] = imageHits[iHit][0]
                    images_fail[iFail][iHit][1] = imageHits[iHit][1]
                    images_fail[iFail][iHit][2] = imageHits[iHit][2]
                    images_fail[iFail][iHit][3] = imageHits[iHit][3]
                iFail += 1
            
    eventsProcessed += 1
    if eventsProcessed % 1000 == 0:
        printLock.acquire()
        print '\tProcessed', eventsProcessed, 'events...'
        printLock.release()

    np.savez_compressed(fileName.split('/')[-1] + '.npz', 
                        images_reco=images_reco,
                        images_fail=images_fail,
                        labels_reco=labels_reco,
                        labels_fail=labels_fail)

    fin.Close()
    semaphore.release()

##############################

iFile = 0
for f in glob.glob('/store/user/bfrancis/images_SingleEle2017F/hist_*.root'):
    outFileName = f.split('/')[-1] + '.npz'
    if os.path.isfile('/store/user/bfrancis/images_SingleEle2017F_pkl/' + outFileName) or os.path.isfile(outFileName):
        continue

    while active_count() > 20:
        time.sleep(1)

    threads.append(Thread(target = processFile, args = (f,)))
    threads[-1].start()

    iFile += 1
    if iFile % 10 == 0:
        printLock.acquire()
        print 'Starting on file:', iFile
        printLock.release()

for thread in threads:
    thread.join()
