import ROOT as r
from ROOT import gROOT
from ROOT.Math import XYZVector
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
import pandas as pd

gROOT.ProcessLine('.L /home/llavezzo/DisappTrksML/TreeMaker/interface/Infos.h+')
gROOT.SetBatch()

dataDir = '/data/disappearingTracks/'
fname = 'original/images_DYJets50.root'
fOut = 'images_DYJets50_norm_40x40.pkl'

##### config params #####
scaling = False
res_eta = 40
res_phi = 40
eta_ub,eta_lb = 0.5,-0.5
phi_ub,phi_lb = 0.5,-0.5
#########################

#import data
fin = r.TFile(dataDir + fname)
tree = fin.Get('trackImageProducer/tree')
print("Added",tree.GetEntries(),"from",fname)

def convert_eta(eta):
    return int(round(((res_eta-1)*1.0/(eta_ub-eta_lb))*(eta-eta_lb)))

def convert_phi(phi):
    return int(round(((res_phi-1)*1.0/(phi_ub-phi_lb))*(phi-phi_lb)))

def type_to_channel(hittype):
    #none
    if(hittype == 0): return -1
    #ECAL (EE,EB)
    if(hittype == 1 or hittype == 2): return 0
    #ES
    if(hittype == 3): return 1 
    #HCAL
    if(hittype == 4): return 2
    #Muon (CSC,DT,RPC)
    if(hittype == 5 or hittype ==6 or hittype == 7): return 3

def check_track(track):
    if(abs(track.genMatchedID)==11 and abs(track.genMatchedDR) < 0.1): return 1
    if(abs(track.genMatchedID)==13 and abs(track.genMatchedDR) < 0.1): return 2
    return 0

def passesSelection(track):

    momentum = XYZVector(track.px, track.py, track.pz)
    eta = momentum.Eta()
    pt = math.sqrt(momentum.Perp2())

    if not abs(eta) < 2.4: return False
    # if not pt > 30: return False
    if track.inGap: return False
    # if not track.trackIso / pt < 0.05: return False
    if not abs(track.dRMinJet) > 0.5: return False
    return True

rows = []

for iEvent,event in enumerate(tree):
    
    if(iEvent%1000==0): print(iEvent)
    
    for iTrack,track in enumerate(event.tracks):

        if(not passesSelection(track)): continue
            
        matrix = np.zeros([res_eta,res_phi,4])

        momentum = XYZVector(track.px,track.py,track.pz)
        track_eta = momentum.Eta()
        track_phi = momentum.Phi()
        
        for iHit,hit in enumerate(event.recHits):
        
            dEta = track_eta - hit.eta
            dPhi = track_phi - hit.phi
            # branch cut [-pi, pi)
            if abs(dPhi) > math.pi:
                dPhi -= round(dPhi / (2. * math.pi)) * 2. * math.pi

            if(dPhi > phi_ub or dPhi < phi_lb): continue
            if(dEta > eta_ub or dEta < eta_lb): continue

            dEta = convert_eta(dEta)
            dPhi = convert_phi(dPhi)

            channel = type_to_channel(hit.detType)
            if(channel == -1): continue

            if channel != 3: matrix[dEta,dPhi,channel] += hit.energy
            else: matrix[dEta][dPhi][channel] += 1

        if(scaling):
            scale = matrix[:,:,:3].max()
            scale_muons = matrix[:,:,3].max()
            if scale > 0: matrix[:,:,:3] = matrix[:,:,:3]*1.0/scale
            if scale_muons > 0: matrix[:,:,3] = matrix[:,:,3]*1.0/scale_muons

        matrix = matrix.flatten().reshape([matrix.shape[0]*matrix.shape[1]*matrix.shape[2],])  
        info = np.array([check_track(track),
            track.deltaRToClosestElectron,
            track.deltaRToClosestMuon,
            track.deltaRToClosestTauHad])

        rows.append(np.concatenate([info,matrix])) 
            
print("Saving to",dataDir)

columns = ['type','deltaRToClosestElectron','deltaRToClosestMuon','deltaRToClosestTau']
pixels = [i for i in range(res_eta*res_phi*4)]
df = pd.DataFrame(rows, columns=np.concatenate([columns,pixels]))
df.to_pickle(dataDir+fOut)