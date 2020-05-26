import ROOT as r
import numpy as np
import os
import matplotlib.pyplot as plt

dataDir = '/mnt/c/users/llave/Documents/CMS/data/'
fname1 = 'orig/images_v2_DYJetsM5to50.root'
fname2 = 'orig/images_v2_DYJetsM50.root'
chain = r.TChain("trackImageProducer/tree")
chain.Add(dataDir+fname1)
chain.Add(dataDir+fname2)
chain.GetEntries()
chain.Merge(dataDir+'merged.root')
del chain

fin = r.TFile(dataDir + 'merged.root')
tree = fin.Get('tree')

energies = []
for event in tree:
    
    for iTrack in range(len(event.track_eta)):

        #truth electrons
        if(abs(event.track_genMatchedID[iTrack])!=11): continue

        #electron events where the RECO failed 
        if(event.track_deltaRToClosestElectron[iTrack] < 0.1 and event.track_deltaRToClosestElectron[iTrack] > -0.1): continue
            
        for iHit in range(len(event.recHits_eta)):
            energies.append(event.recHits_energy[iHit])
        
from sklearn import preprocessing
newenergies = np.array(energies)
newenergies = newenergies.reshape(-1,1)
pt = preprocessing.PowerTransformer(method='yeo-johnson', standardize=False)
newenergies = pt.fit_transform(newenergies)

import math

nevents = 0
count = 0
e_events,bkg_events = [],[]
res_eta = 20
res_phi = 20

#eta and phi upper and lower bounds
eta_ub,eta_lb = 3,-3
phi_ub,phi_lb = math.pi,-math.pi

def convert_eta(eta):
    return int(round(((res_eta-1)/(eta_ub-eta_lb))*(eta-eta_lb)))

def convert_phi(phi):
    return int(round(((res_phi-1)/(phi_ub-phi_lb))*(phi-phi_lb)))

#overlap EE and EB
def type_to_channel(hittype):
    if(hittype == 0 or hittype == 1): return (hittype)
    else: return (hittype-1)
    
for event in tree:
               
    matrix = np.zeros([res_eta,res_phi,4])
    
    for iTrack in range(len(event.track_eta)):
        
        #truth electrons
        if(abs(event.track_genMatchedID[iTrack])!=11): continue

        #electron events where the RECO failed 
        if(event.track_deltaRToClosestElectron[iTrack] < 0.1 and event.track_deltaRToClosestElectron[iTrack] > -0.1): continue
            
        nevents+=1
        
        for iHit in range(len(event.recHits_eta)):

            dEta = event.track_eta[iTrack] - event.recHits_eta[iHit]
            dPhi = event.track_phi[iTrack] - event.recHits_phi[iHit]
            # branch cut [-pi, pi)
            if abs(dPhi) > math.pi:
                dPhi -= round(dPhi / (2. * math.pi)) * 2. * math.pi

            if(dPhi > phi_ub or dPhi < phi_lb): continue
            if(dEta > eta_ub or dEta < eta_lb): continue

            dEta = convert_eta(dEta)
            dPhi = convert_phi(dPhi)

            #matrix[dEta,dPhi,track.recHits_detType[iHit]] = track.recHits_energy[iHit]
            channel = type_to_channel(event.recHits_detType[iHit])
            matrix[dEta,dPhi,channel] = newenergies[count]
            count+=1

        
        if(abs(event.track_genMatchedDR[iTrack]) < 0.1): e_events.append(matrix)
        else: bkg_events.append(matrix)

print(len(e_events),len(bkg_events))
np.save(dataDir+'e_DYJets50V3_norm_20x20', e_events)
np.save(dataDir+'bkg_DYJets50V3_norm_20x20', bkg_events)
