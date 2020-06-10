import ROOT as r
from ROOT import gROOT
from ROOT.Math import XYZVector
import numpy as np
import os
import matplotlib.pyplot as plt
import math

gROOT.ProcessLine('.L /home/llavezzo/DisappTrksML/TreeMaker/interface/Infos.h+')
gROOT.SetBatch()

dataDir = '/data/disappearingTracks/'
fname = 'original/images_DYJets50.root'
tag = '_DYJets50_norm_40x40'

##### config params #####
scaling = True
res_eta = 40
res_phi = 40
#########################

#import data
fin = r.TFile(dataDir + fname)
tree = fin.Get('trackImageProducer/tree')
print("Added",tree.GetEntries(),"from",fname)

#export electrons, muons, and everything else
#image of hit and reco results
e_events,m_events,bkg_events = [],[],[]
e_reco,m_reco,bkg_reco = [],[],[]

#eta and phi upper and lower bounds
eta_ub,eta_lb = 3,-3
phi_ub,phi_lb = math.pi,-math.pi

def convert_eta(eta):
    return int(round(((res_eta-1)/(eta_ub-eta_lb))*(eta-eta_lb)))

def convert_phi(phi):
    return int(round(((res_phi-1)/(phi_ub-phi_lb))*(phi-phi_lb)))

def type_to_channel(hittype):
    #none
    if(hittype == 0 ): return -1
    #ECAL (EE,EB)
    if(hittype == 1 or hittype == 2): return 0
    #ES
    if(hittype == 3): return 1
    #HCAL
    if(hittype == 4): return 2
    #Muon (CSC,DT,RPC)
    if(hittype == 5 or hittype ==6 or hittype == 7): return 3


for iEvent,event in enumerate(tree):
    
    if(iEvent%1000==0): print(iEvent)
    
    for iTrack,track in enumerate(event.tracks):
              
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

                matrix[dEta,dPhi,channel] += hit.energy if channel != 3 else 1

            if(scaling):
                scale = matrix[:,:,:3].max()
                scale_muons = matrix[:,:,3].max()
                if scale > 0: matrix[:,:,:3] = matrix[:,:,:3]*1.0/scale
                if scale_muons > 0: matrix[:,:,3] = matrix[:,:,3]*1.0/scale_muons

            #truth electrons
            if(abs(track.genMatchedID)==11 and abs(track.genMatchedDR) < 0.1):
                e_events.append(matrix)
                e_reco.append([track.deltaRToClosestElectron,track.deltaRToClosestMuon,track.deltaRToClosestTauHad])
            #truth muons
            elif(abs(track.genMatchedID)==13 and abs(track.genMatchedDR) < 0.1):
                m_events.append(matrix)
                m_reco.append([track.deltaRToClosestElectron,track.deltaRToClosestMuon,track.deltaRToClosestTauHad])
            #everything else
            else:
                bkg_events.append(matrix)
                bkg_reco.append([track.deltaRToClosestElectron,track.deltaRToClosestMuon,track.deltaRToClosestTauHad])

            
print("Saving to",dataDir)
print(len(e_events),"electron events,",len(m_events),"muon events, and",len(bkg_events),"background events")
np.save(dataDir+'e'+tag, e_events)
np.save(dataDir+'muon'+tag, m_events)
np.save(dataDir+'bkg'+tag, bkg_events)
np.save(dataDir+'e_reco'+tag, e_reco)
np.save(dataDir+'muon_reco'+tag, m_reco)
np.save(dataDir+'bkg_reco'+tag, bkg_reco)