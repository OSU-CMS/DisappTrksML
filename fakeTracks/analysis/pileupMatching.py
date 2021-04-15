#!/usr/bin/env python

import os
import sys
import math
import glob
import pickle
import ROOT as r
from ROOT import TFile, TTree, gROOT, TH1F
from ROOT.Math import XYZVector
import numpy as np
import time

dataDir = '/store/user/mcarrigan/fakeTracks/selection_v7p1_aMCNLO/'

#gROOT.ProcessLine('.L Infos2.h++')

h_dz_real = r.TH1F("h_dz_real", "Real Track dz Matching", 1000, 0, 5)
h_dz_fake = r.TH1F("h_dz_fake", "Fake Track dz Matching", 1000, 0, 5)

outFile = r.TFile("dzMatchingPlots.root", "recreate")
outFile.cd()

def pileupMatching(track):
    min_dz = 10e6
    for vertex in event.pileupZPosition:
        dZ = abs(track.vz - vertex)
        if dZ < min_dz: min_dz = dZ
    if(min_dz == 10e6): 
        min_dz = -1
        print("No pileup vertices")
    return min_dz

file_count = 0
for filename in os.listdir(dataDir):
    #if 'images' or 'root' not in filename: continue
    #if file_count > 10: continue
    print ("File "+dataDir+filename)

    fin = TFile(dataDir+filename, 'read')
    fakeTree = fin.Get('fakeTree')
    realTree = fin.Get('realTree')
    print("Fake Tree Events: " + str(fakeTree.GetEntries()))
    print("Real Tree Events: " + str(realTree.GetEntries()))

    for class_label,tree in zip([0,1],[realTree,fakeTree]):
        for iEvent, event in enumerate(tree):
            for iTrack, track in enumerate(event.tracks):
                dZ = pileupMatching(track)
                if(class_label == 0): h_dz_real.Fill(dZ)
                if(class_label == 1): h_dz_fake.Fill(dZ)
    file_count += 1

outFile.cd()            
h_dz_real.Write("dz_Real")
h_dz_fake.Write("dz_Fake")
outFile.Close() 
           

