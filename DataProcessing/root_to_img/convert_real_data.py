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
import sys


gROOT.ProcessLine('.L Infos.h++')
gROOT.SetBatch()

# script arguments
process = int(sys.argv[1])
print("Process",process)

# name of the file to import
try:
    files = np.load('fileslist.npy')
    fileNum = files[process]
except:
    fileNum = process
fname = "hist_"+str(fileNum)+".root"

# output file tag
fOut = '0p25_'+str(fileNum)

##### config params #####
scaling = False
tanh_scaling = False
res_eta = 40
res_phi = 40
eta_ub,eta_lb = 0.25,-0.25
phi_ub,phi_lb = 0.25,-0.25
#########################

# import data
dataDir = '/store/user/bfrancis/images_SingleEle2017F/'
fin = r.TFile(dataDir + fname)
tree = fin.Get('trackImageProducer/tree')
print("Opened file",fname)
nEvents = int(tree.GetEntries())
if(nEvents == 0):
    sys.exit("0 events found in file")
print("Added",nEvents)

# Convert coordinates from original mapping
# to range as specified in parameters
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

# Match electrons, muons
def check_track(track):
    if(track.isTagProbeElectron == 1): return 1
    else: return 0

def passesIsolatedTrackSelection(track):

    momentum = XYZVector(track.px, track.py, track.pz)
    eta = momentum.Eta()
    pt = math.sqrt(momentum.Perp2())

    if not abs(eta) < 2.4: return False
    if not pt > 30: return False
    if track.inGap: return False
    if not track.nValidPixelHits >= 4: return False
    if not track.nValidHits >= 4: return False
    if not track.missingInnerHits == 0: return False
    if not track.missingMiddleHits == 0: return False
    if not track.trackIso / pt < 0.05: return False
    if not abs(track.d0) < 0.02: return False
    if not abs(track.dz) < 0.5: return False
    if not abs(track.dRMinJet) > 0.5: return False
    if not abs(track.deltaRToClosestElectron) > 0.15: return False
    if not abs(track.deltaRToClosestMuon) > 0.15: return False
    if not abs(track.deltaRToClosestTauHad) > 0.15: return False
    return True

# images and infos split by gen matched type
images = [[],[]]
infos = [[],[]]
ID = 0

for iEvent,event in enumerate(tree):
    
    if(iEvent%1000==0): print(iEvent)

    nPV = event.nPV
        
    for iTrack,track in enumerate(event.tracks):

        if(not passesIsolatedTrackSelection(track)): continue
        
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

        # scaling options
        if(scaling):
            scale = matrix[:,:,:3].max()
            scale_muons = matrix[:,:,3].max()
            if scale > 0: matrix[:,:,:3] = matrix[:,:,:3]*1.0/scale
            if scale_muons > 0: matrix[:,:,3] = matrix[:,:,3]*1.0/scale_muons
        if(tanh_scaling):
            matrix = np.tanh(matrix)

        matrix = matrix.flatten().reshape([matrix.shape[0]*matrix.shape[1]*matrix.shape[2],])  
        matrix = matrix.astype('float32')
        matrix = np.append(ID,matrix)
        
        isProbe = check_track(track)
        info = np.array([
            fileNum,
            ID,
            isProbe,
            nPV,
            track.deltaRToClosestElectron,
            track.deltaRToClosestMuon,
            track.deltaRToClosestTauHad,
            track_eta,
            track_phi,
            track.genMatchedID,
            track.genMatchedDR])

        images[isProbe].append(matrix)
        infos[isProbe].append(info)

        ID+=1
            

# check for errors before saving
nEvents = 0
for i in range(2):
    if(len(images[i])!=len(infos[i])): sys.exit("Images and infos don't match!")
    nEvents += len(images[i])
if(nEvents == 0): sys.exit("The output file is empty")

print("Saving to",fOut)
np.savez_compressed("images_bkg_"+fOut,images=images[0],infos=infos[0])
np.savez_compressed("images_e_"+fOut,images=images[1],infos=infos[1])
