import matplotlib.pyplot as plt
import ROOT as r
import math
import numpy as np
from ROOT import gROOT
from ROOT.Math import XYZVector
import os
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import sys

gROOT.ProcessLine('.L DisappTrksML/TreeMaker/interface/Infos.h++')
gROOT.SetBatch()

imageDir = '/store/user/bfrancis/images_v5/DYJetsToLL_crab/'

fin = r.TFile(imageDir + 'images_61.root')
fout = r.TFile('analysisPlots.root', 'RECREATE')
tree = fin.Get('trackImageProducer/tree')
#print("Opened file",fname)
nEvents = tree.GetEntries()
if(nEvents == 0):
    sys.exit("0 events found in file")
print("Added",nEvents)

fakeCount = 0

hist_dR = r.TH1F("histdR", "histdR", 50, 0, 2)
c1 = r.TCanvas("canvas1", "canvas1", 800, 600)

for iEvent, event in enumerate(tree):
    if(iEvent%1000==0): print(iEvent)

    nPV = event.nPV

    for iTrack,track in enumerate(event.tracks):
        min_dR = 10e7
        print(iEvent, iTrack, track.eta, track.phi)
        for iGen,genParticle in enumerate(event.genParticles):
            dR = np.sqrt(abs(track.eta - genParticle.eta)**2 + abs(track.phi - genParticle.phi)**2)
            if dR < min_dR: min_dR = dR
        
        if min_dR > 0.3:
            print(iTrack, min_dR)
            fakeCount += 1
        hist_dR.Fill(min_dR)

#c1.cd()
#hist_dR.Draw()
#c1.SaveAs("dR_matching.png")
hist_dR.Write()
fout.Close()













