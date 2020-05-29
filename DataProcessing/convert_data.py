import ROOT as r
import numpy as np
import os
import matplotlib.pyplot as plt
import math

dataDir = '/mnt/c/users/llave/Documents/CMS/data/'
fname = 'orig/images_v3_DYJets50.root'
fin = r.TFile(dataDir + fname)
tree = fin.Get('trackImageProducer/tree')

e,bkg = 0,0
e_events,bkg_events = [],[]
e_reco_results,bkg_reco_results = [],[]
reco_results = []
res_eta = 40
res_phi = 40

#eta and phi upper and lower bounds
eta_ub,eta_lb = 3,-3
phi_ub,phi_lb = math.pi,-math.pi

def convert_eta(eta):
    return int(round(((res_eta-1)/(eta_ub-eta_lb))*(eta-eta_lb)))

def convert_phi(phi):
    return int(round(((res_phi-1)/(phi_ub-phi_lb))*(phi-phi_lb)))

#overlap (EE,EB), (CSC,DT,RPC)
def type_to_channel(hittype):
    #none
    if(hittype == 0 ): return 0
    #(EE,EB)
    if(hittype == 1 or hittype == 2): return 1
    #ES
    if(hittype == 3): return 2
    #HCAL
    if(hittype == 4): return 3
    #(CSC,DT,RPC)
    if(hittype == 5 or hittype ==6 or hittype == 7): return 4

for i,event in enumerate(tree):
    
    if(i%1000==0): 
        print(i)

    for iTrack in range(len(event.track_eta)):

            matrix = np.zeros([res_eta,res_phi,5])

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
                channel = type_to_channel(event.recHits_detType[iHit])

                matrix[dEta,dPhi,channel] += event.recHits_energy[iHit]
                
            scale = matrix.max()
            matrix = matrix*1/scale

            #truth electrons
            if(abs(event.track_genMatchedID[iTrack])==11 and abs(event.track_genMatchedDR[iTrack]) < 0.1):
                e_events.append(matrix)
                if(abs(event.track_deltaRToClosestElectron[iTrack])<0.15): e_reco_results.append(1)
                else: e_reco_results.append(0)

            #everything else
            else:
                bkg_events.append(matrix)
                if(abs(event.track_deltaRToClosestElectron[iTrack])<0.15): bkg_reco_results.append(1)
                else: bkg_reco_results.append(0)

            
print(len(e_events),len(bkg_events),len(reco_results))
np.save(dataDir+'e_DYJets50V3_norm_40x40', e_events)
np.save(dataDir+'bkg_DYJets50V3_norm_40x40', bkg_events)
np.save(dataDir+'e_reco_DYJets50V3_norm_40x40',e_reco_results)
np.save(dataDir+'bkg_reco_DYJets50V3_norm_40x40',bkg_reco_results)