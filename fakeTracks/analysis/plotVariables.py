import numpy as np
import ROOT as r
import os
import sys


#Suppress plots from printing to screen
r.gROOT.SetBatch(1)

#dyData = ['/store/user/mcarrigan/fakeTracks/converted_madgraph_fullSelection_4PlusLayer_v7p1/', '/store/user/mcarrigan/fakeTracks/converted_aMC_fullSelection_4PlusLayer_v7p1/']
dyData = ['/store/user/mcarrigan/fakeTracks/selection_v9_DYJets_aMCNLO/']

signalData = ['/store/user/mcarrigan/fakeTracks/converted_higgsino_700_10_4PlusLayer_v7p1/', '/store/user/mcarrigan/fakeTracks/converted_higgsino_700_100_4PlusLayer_v7p1/',
              '/store/user/mcarrigan/fakeTracks/converted_higgsino_700_1000_4PlusLayer_v7p1/', '/store/user/mcarrigan/fakeTracks/converted_higgsino_700_10000_4PlusLayer_v7p1/']

neutrinoGunData = ['/store/user/mcarrigan/fakeTracks/selection_v8_NeutrinoGun_ext/']

zeroBiasData = ['/store/user/mcarrigan/fakeTracks/selection_ZeroBias_2017D_v8/']

#trackIso, eta, phi, nPV, dRMinJet, ecalo, pt, d0, dz, charge, nValidPixelHits, nValidHits, missingOuterHits, dEdxPixel, dEdxStrip, numMeasurementsPixel,
                          #numMeasurementsStrip, numSatMeasurementsPixel, numSatMeasurementsStrip]

out = r.TFile("plottedVariables.root", "recreate")

layers = -1

Debug = True

varDict = {'trackIso':0, 'eta':1, 'phi':2, 'nPV':3, 'drMinJet':4, 'ecalo':5, 'pt':6, 'd0':7, 'dz':8, 'charge':9, 'pixelHits':10, 'hits':11, 'missingOuter':12, 'dEdxPixel':13, 'dEdxStrip':14, 'pixelMeasurements':15, 'stripMeasurements':16, 'pixelSat':17, 'stripSat':18, 'sumEnergy':19, 'diffEnergy':20, 'dz1':21, 'd01':22, 'dz2':23, 'd02':24, 'dz3':25, 'd03':26, 'layer1':27, 'charge1':28, 'subDet1':29, 'pixelHitSize1':30, 'pixelHitSizeX1':31, 'pixelHitSizeY1':32, 'stripSelection1':33, 'hitPosX1':34, 'hitPosY1':35, 'layer2':36, 'charge2':37, 'subDet2':38, 'pixelHitSize2':39, 'pixelHitSizeX2':40, 'pixelHitSizeY2':41, 'stripSelection2':42, 'hitPosX2':43, 'hitPosY2':44, 'layer3':45, 'charge3':46, 'subDet3':47, 'pixelHitSize3':48, 'pixelHitSizeX3':49, 'pixelHitSizeY3':50, 'stripSelection3':51, 'hitPosX3':52, 'hitPosY3':53, 'layer4':54, 'charge4':55, 'subDet4':56, 'pixelHitSize4':57, 'pixelHitSizeX4':58, 'pixelHitSizeY4':59, 'stripSelection4':60, 'hitPosX4':61, 'hitPosY4':62, 'layer5':63, 'charge5':64, 'subDet5':65, 'pixelHitSize5':66, 'pixelHitSizeX5':67, 'pixelHitSizeY5':68, 'stripSelection5':69, 'hitPosX5':70, 'hitPosY5':71, 'layer6':72, 'charge6':73, 'subDet6':74, 'pixelHitSize6':75, 'pixelHitSizeX6':76, 'pixelHitSizeY6':77, 'stripSelection6':78, 'hitPosX6':79, 'hitPosY6':80}

h_trackIso = r.TH1F("h_trackIso", "Track Iso", 100, 0, 50) 
h_eta = r.TH1F("h_eta", "Eta", 50, -2.5, 2.5)
h_phi = r.TH1F("h_phi", "Phi", 100, -4, 4)
h_nPV = r.TH1F("h_nPV", "nPV", 125, 0, 125)
h_drMinJet = r.TH1F("h_drMinJet", "dRMinJet", 110, -1, 10)
h_ecalo = r.TH1F("h_ecalo", "Ecalo", 1000, 0, 1000)
h_pt = r.TH1F("h_pt", "PT", 1000, 0, 1000)
h_d0 = r.TH1F("h_d0", "d0", 100, -0.1, 0.1)
h_dz = r.TH1F("h_dz", "dz", 50, -1, 1)
h_charge = r.TH1F("h_charge", "Charge", 2, -1, 1)
h_pixelHits = r.TH1F("h_pixelHits", "Valid Pixel Hits", 15, 0, 15)
h_Hits = r.TH1F("h_Hits", "Valid Hits", 40, 0, 40)
h_missingOuter = r.TH1F("h_missingOuter", "Number of Missing Outer Hits", 16, 0, 16)
h_dEdxPixel = r.TH1F("h_dEdxPixel", "dEdx Pixel", 100, 0, 10)
h_dEdxStrip = r.TH1F("h_dEdxStrip", "dEdx Strip", 100, 0, 10)
h_pixelMeasurements = r.TH1F("h_pixelMeasurements", "Number of Pixel Measurements", 15, 0, 15)
h_stripMeasurements = r.TH1F("h_stripMeasurements", "Number of Strip Measurements", 30, 0, 30)
h_pixelSat = r.TH1F("h_pixelSat", "Number of Saturated Pixel Measurements", 10, 0, 10)
h_stripSat = r.TH1F("h_stripSat", "Number of Saturated Strip Meausrements", 15, 0, 15)
h_sumEnergy = r.TH1F('h_sumEnergy', "Sum of Layer Energies", 1000, 0, 200)
h_diffEnergy = r.TH1F('h_diffEnergy', "Max Difference Between Layer Energies", 500, 0, 50)
h_dz1 = r.TH1F("h_dz1", "Track dz to Closest Vertex", 50, 0, 2)
h_d01 = r.TH1F("h_d01", "Track d0 to Closest Vertex", 100, 0, 0.2)
h_dz2 = r.TH1F("h_dz2", "Track dz to 2nd Closest Vertex", 50, 0, 2)
h_d02 = r.TH1F("h_d02", "Track d0 to 2nd Closest Vertex", 100, 0, 0.2)
h_dz3 = r.TH1F("h_dz3", "Track dz to 3rd Closest Vertex", 50, 0, 2)
h_d03 = r.TH1F("h_d03", "Track d0 to 3rd Closest Vertex", 100, 0, 0.2)

h_layer1 = r.TH1F("h_layer1", "Layer", 10, 0, 10)
h_charge1 = r.TH1F("h_charge1", "Charge", 200, 0, 200)
h_subDet1 = r.TH1F("h_subDet1", "SubDetector", 20, -10, 10)
h_pixelHitSize1 = r.TH1F("h_pixelHitSize1", "Pixel Hit Size", 21, -1, 20)
h_pixelHitSizeX1 = r.TH1F("h_pixelHitSizeX1", "Pixel Hit Size X", 11, -1, 10)
h_pixelHitSizeY1 = r.TH1F("h_pixelHitSizeY1", "Pixel Hit Size Y", 16, -1, 15)
h_stripSelection1 = r.TH1F("h_stripSelection1", "Strip Selection", 2, 0, 2)
h_hitPosX1 = r.TH1F("h_hitPosX1", "Hit Position X", 100, -10, 10)
h_hitPosY1 = r.TH1F("h_hitPosY1", "Hit Position Y", 200, -20, 20)

h_layer2 = r.TH1F("h_layer2", "Layer", 10, 0, 10)
h_charge2 = r.TH1F("h_charge2", "Charge", 200, 0, 200)
h_subDet2 = r.TH1F("h_subDet2", "SubDetector", 20, -10, 10)
h_pixelHitSize2 = r.TH1F("h_pixelHitSize2", "Pixel Hit Size", 21, -1, 20)
h_pixelHitSizeX2 = r.TH1F("h_pixelHitSizeX2", "Pixel Hit Size X", 11, -1, 10)
h_pixelHitSizeY2 = r.TH1F("h_pixelHitSizeY2", "Pixel Hit Size Y", 16, -1, 15)
h_stripSelection2 = r.TH1F("h_stripSelection2", "Strip Selection", 2, 0, 2)
h_hitPosX2 = r.TH1F("h_hitPosX2", "Hit Position X", 100, -10, 10)
h_hitPosY2 = r.TH1F("h_hitPosY2", "Hit Position Y", 200, -20, 20)

h_layer3 = r.TH1F("h_layer3", "Layer", 10, 0, 10)
h_charge3 = r.TH1F("h_charge3", "Charge", 200, 0, 200)
h_subDet3 = r.TH1F("h_subDet3", "SubDetector", 20, -10, 10)
h_pixelHitSize3 = r.TH1F("h_pixelHitSize3", "Pixel Hit Size", 21, -1, 20)
h_pixelHitSizeX3 = r.TH1F("h_pixelHitSizeX3", "Pixel Hit Size X", 11, -1, 10)
h_pixelHitSizeY3 = r.TH1F("h_pixelHitSizeY3", "Pixel Hit Size Y", 16, -1, 15)
h_stripSelection3 = r.TH1F("h_stripSelection3", "Strip Selection", 2, 0, 2)
h_hitPosX3 = r.TH1F("h_hitPosX3", "Hit Position X", 100, -10, 10)
h_hitPosY3 = r.TH1F("h_hitPosY3", "Hit Position Y", 200, -20, 20)

h_layer4 = r.TH1F("h_layer4", "Layer", 10, 0, 10)
h_charge4 = r.TH1F("h_charge4", "Charge", 200, 0, 200)
h_subDet4 = r.TH1F("h_subDet4", "SubDetector", 20, -10, 10)
h_pixelHitSize4 = r.TH1F("h_pixelHitSize4", "Pixel Hit Size", 21, -1, 20)
h_pixelHitSizeX4 = r.TH1F("h_pixelHitSizeX4", "Pixel Hit Size X", 11, -1, 10)
h_pixelHitSizeY4 = r.TH1F("h_pixelHitSizeY4", "Pixel Hit Size Y", 16, -1, 15)
h_stripSelection4 = r.TH1F("h_stripSelection4", "Strip Selection", 2, 0, 2)
h_hitPosX4 = r.TH1F("h_hitPosX4", "Hit Position X", 100, -10, 10)
h_hitPosY4 = r.TH1F("h_hitPosY4", "Hit Position Y", 200, -20, 20)

h_layer5 = r.TH1F("h_layer5", "Layer", 10, 0, 10)
h_charge5 = r.TH1F("h_charge5", "Charge", 200, 0, 200)
h_subDet5 = r.TH1F("h_subDet5", "SubDetector", 20, -10, 10)
h_pixelHitSize5 = r.TH1F("h_pixelHitSize5", "Pixel Hit Size", 21, -1, 20)
h_pixelHitSizeX5 = r.TH1F("h_pixelHitSizeX5", "Pixel Hit Size X", 11, -1, 10)
h_pixelHitSizeY5 = r.TH1F("h_pixelHitSizeY5", "Pixel Hit Size Y", 16, -1, 15)
h_stripSelection5 = r.TH1F("h_stripSelection5", "Strip Selection", 2, 0, 2)
h_hitPosX5 = r.TH1F("h_hitPosX5", "Hit Position X", 100, -10, 10)
h_hitPosY5 = r.TH1F("h_hitPosY5", "Hit Position Y", 200, -20, 20)

h_layer6 = r.TH1F("h_layer6", "Layer", 10, 0, 10)
h_charge6 = r.TH1F("h_charge6", "Charge", 200, 0, 200)
h_subDet6 = r.TH1F("h_subDet6", "SubDetector", 20, -10, 10)
h_pixelHitSize6 = r.TH1F("h_pixelHitSize6", "Pixel Hit Size", 21, -1, 20)
h_pixelHitSizeX6 = r.TH1F("h_pixelHitSizeX6", "Pixel Hit Size X", 11, -1, 10)
h_pixelHitSizeY6 = r.TH1F("h_pixelHitSizeY6", "Pixel Hit Size Y", 16, -1, 15)
h_stripSelection6 = r.TH1F("h_stripSelection6", "Strip Selection", 2, 0, 2)
h_hitPosX6 = r.TH1F("h_hitPosX6", "Hit Position X", 100, -10, 10)
h_hitPosY6 = r.TH1F("h_hitPosY6", "Hit Position Y", 200, -20, 20)

h_layers = [h_layer1, h_layer2, h_layer3, h_layer4, h_layer5,h_layer6]
h_charges = [h_charge1, h_charge2, h_charge3, h_charge4, h_charge5, h_charge6]
h_subDets = [h_subDet1, h_subDet2, h_subDet3, h_subDet4, h_subDet5, h_subDet6]
h_pixelHitSizes = [h_pixelHitSize1, h_pixelHitSize2, h_pixelHitSize3, h_pixelHitSize4, h_pixelHitSize5, h_pixelHitSize6]
h_pixelHitSizesX = [h_pixelHitSizeX1, h_pixelHitSizeX2, h_pixelHitSizeX3, h_pixelHitSizeX4, h_pixelHitSizeX5, h_pixelHitSizeX6]
h_pixelHitSizesY = [h_pixelHitSizeY1, h_pixelHitSizeY2, h_pixelHitSizeY3, h_pixelHitSizeY4, h_pixelHitSizeY5, h_pixelHitSizeY6]
h_stripSelections = [h_stripSelection1, h_stripSelection2, h_stripSelection3, h_stripSelection4, h_stripSelection5, h_stripSelection6]
h_hitPosXs = [h_hitPosX1, h_hitPosX2, h_hitPosX3, h_hitPosX4, h_hitPosX5, h_hitPosX6]
h_hitPosYs = [h_hitPosY1, h_hitPosY2, h_hitPosY3, h_hitPosY4, h_hitPosY5, h_hitPosY6]

def trackMatching(track, event):
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

def makePUScaleFactors(dataPU, mcPU):
    f_data = r.TFile(dataPU)
    f_mc = r.TFile(mcPU)

    h_data = r.TH1D(f_data.Get('pileup'))
    h_mc = r.TH1D(f_mc.Get('mc2017_12Apr2018'))

    scale_factors = []

    for bin in range(h_mc.GetNbinsX()):
        mc_bin = h_mc.GetBinContent(bin) / h_mc.Integral()
        data_bin = -1
        if bin <= h_data.GetNbinsX():
            data_bin = h_data.GetBinContent(bin) / h_data.Integral()
        sf = 1
        if(data_bin == -1):
            sf = -1
        elif(data_bin == 0 and mc_bin == 0):
            sf = 1
        else:
            sf = float(data_bin / mc_bin)
        
        scale_factors.append(sf)
        
        print("bin:" , bin, "mc_bin:", mc_bin, "data_bin:", data_bin, "weight:", sf)

    return scale_factors

def getScaleFactor(scale_factors, numPV):
    return scale_factors[numPV]

def plotDatasets(datasets, trees, track_names, dataType):
    
    trackNameCounter = 0

    for idataSet, dataset in enumerate(datasets):
        #print("Looking at dataset: " + dataset)

        for iTree, tree in enumerate(trees):

            fileCount = 0
            mychain = r.TChain(tree)
            if(Debug == True):
                mychain.Add(dataset + "hist_8*.root")
            else: mychain.Add(dataset+"hist*.root")
            #for filename in os.listdir(dataset):
            if(mychain.GetEntries() > 0):

                print("Number of entries in dataset " + dataset + " " + str(mychain.GetEntries()))
                #if 'root' not in filename: continue
                #if(Debug == True):
                #    if fileCount > 21: break
                #fin = r.TFile(dataset + filename, "read")
                #mytree = fin.Get(tree)
                fileCount += 1
                #print("Looking at file: " + filename + ", tree: " +  tree + ", with " + str(mytree.GetEntries()) + " entries.")
                trackCount = 0
                for event in mychain:

                    plotEvent = False

                    weight = 1
                    if(dataType[idataSet] == 'mc'):
                        numTruePV = event.numTruePV
                        scale_factors = PU_dict[dataset]
                        weight = getScaleFactor(scale_factors, numTruePV)

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

                        trackCount += 1
                        plotEvent = True

                        h_trackIso.Fill(track.trackIso, weight)
                        h_eta.Fill(track.eta, weight)
                        h_phi.Fill(track.phi, weight)
                        h_drMinJet.Fill(track.dRMinJet,weight)
                        h_ecalo.Fill(track.ecalo, weight)
                        h_pt.Fill(track.pt, weight)
                        h_d0.Fill(track.d0, weight)
                        h_dz.Fill(track.dz, weight)
                        h_charge.Fill(track.charge, weight)
                        h_pixelHits.Fill(track.nValidPixelHits, weight)
                        h_Hits.Fill(track.nValidHits, weight)
                        h_missingOuter.Fill(track.missingOuterHits, weight)
                        h_dEdxPixel.Fill(track.dEdxPixel, weight)
                        h_dEdxStrip.Fill(track.dEdxStrip, weight)
                        h_pixelMeasurements.Fill(track.numMeasurementsPixel, weight)
                        h_stripMeasurements.Fill(track.numMeasurementsStrip, weight)
                        h_pixelSat.Fill(track.numSatMeasurementsPixel, weight)
                        h_stripSat.Fill(track.numSatMeasurementsStrip, weight)

                        if(layers == 6 or layers == -1): nLayers = np.zeros((16, 9))
                        else: nLayers = np.zeros((layers, 9))
                        nLayerCount = 0

                        max_energy, min_energy, sum_energy = 0, 10e6, 0

                        for iHit, hit in enumerate(track.dEdxInfo):

                            layerHits = []
                            #print("Event: " + str(iEvent) + ", Track: " + str(iTrack) + " isFake: " + tree + ", subDet: " + str(hit.subDet) + " Layer: "
                            #    + str(hit.hitLayerId), "nLayersWithMeasurement: " + str(nLayersWithMeasurement) + " Eta: " + str(track.eta) + " missingInnerHits: "
                            #    + str(track.missingInnerHits) + " missingMiddleHits: " + str(track.missingMiddleHits) + " missingOuterHits: " + str(track.missingOuterHits) +
                            #    " Charge: " + str(hit.charge))
                            if(hit.hitLayerId < 0): continue
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
                            

                        # Now fill plots that depend on layer/detector
                        h_sumEnergy.Fill(sum_energy, weight)
                        h_diffEnergy.Fill(max_energy-min_energy, weight)

                        v_dz, v_d0 = trackMatching(track, event)

                        h_dz1.Fill(v_d0[0], weight)
                        h_d01.Fill(v_d0[1], weight)
                        h_dz2.Fill(v_d0[2], weight)
                        h_d02.Fill(v_dz[0], weight)
                        h_dz3.Fill(v_dz[1], weight)
                        h_d03.Fill(v_dz[2], weight)

                        h_layers[0].Fill(nLayers[0,0], weight)
                        h_charges[0].Fill(nLayers[0,1], weight)
                        h_subDets[0].Fill(nLayers[0,2], weight)
                        h_pixelHitSizes[0].Fill(nLayers[0,3], weight)
                        h_pixelHitSizesX[0].Fill(nLayers[0,4], weight)
                        h_pixelHitSizesY[0].Fill(nLayers[0,5], weight)
                        h_stripSelections[0].Fill(nLayers[0,6], weight)
                        h_hitPosXs[0].Fill(nLayers[0,7], weight)
                        h_hitPosYs[0].Fill(nLayers[0,8], weight)
                        h_layers[1].Fill(nLayers[1,0], weight)
                        h_charges[1].Fill(nLayers[1,1], weight)
                        h_subDets[1].Fill(nLayers[1,2], weight)
                        h_pixelHitSizes[1].Fill(nLayers[1,3], weight)
                        h_pixelHitSizesX[1].Fill(nLayers[1,4], weight)
                        h_pixelHitSizesY[1].Fill(nLayers[1,5], weight)
                        h_stripSelections[1].Fill(nLayers[1,6], weight)
                        h_hitPosXs[1].Fill(nLayers[1,7], weight)
                        h_hitPosYs[1].Fill(nLayers[1,8], weight)
                        h_layers[2].Fill(nLayers[2,0], weight)
                        h_charges[2].Fill(nLayers[2,1], weight)
                        h_subDets[2].Fill(nLayers[2,2], weight)
                        h_pixelHitSizes[2].Fill(nLayers[2,3], weight)
                        h_pixelHitSizesX[2].Fill(nLayers[2,4], weight)
                        h_pixelHitSizesY[2].Fill(nLayers[2,5], weight)
                        h_stripSelections[2].Fill(nLayers[2,6], weight)
                        h_hitPosXs[2].Fill(nLayers[2,7], weight)
                        h_hitPosYs[2].Fill(nLayers[2,8], weight)
                        h_layers[3].Fill(nLayers[3,0], weight)
                        h_charges[3].Fill(nLayers[3,1], weight)
                        h_subDets[3].Fill(nLayers[3,2], weight)
                        h_pixelHitSizes[3].Fill(nLayers[3,3], weight)
                        h_pixelHitSizesX[3].Fill(nLayers[3,4], weight)
                        h_pixelHitSizesY[3].Fill(nLayers[3,5], weight)
                        h_stripSelections[3].Fill(nLayers[3,6], weight)
                        h_hitPosXs[3].Fill(nLayers[3,7], weight)
                        h_hitPosYs[3].Fill(nLayers[3,8], weight)
                        if(layers < 5): continue
                        h_layers[4].Fill(nLayers[4,0], weight)
                        h_charges[4].Fill(nLayers[4,1], weight)
                        h_subDets[4].Fill(nLayers[4,2], weight)
                        h_pixelHitSizes[4].Fill(nLayers[4,3], weight)
                        h_pixelHitSizesX[4].Fill(nLayers[4,4], weight)
                        h_pixelHitSizesY[4].Fill(nLayers[4,5], weight)
                        h_stripSelections[4].Fill(nLayers[4,6], weight)
                        h_hitPosXs[4].Fill(nLayers[4,7], weight)
                        h_hitPosYs[4].Fill(nLayers[4,8], weight)
                        if(layers < 6): continue
                        h_layers[5].Fill(nLayers[5,0], weight)
                        h_charges[5].Fill(nLayers[5,1], weight)
                        h_subDets[5].Fill(nLayers[5,2], weight)
                        h_pixelHitSizes[5].Fill(nLayers[5,3], weight)
                        h_pixelHitSizesX[5].Fill(nLayers[5,4], weight)
                        h_pixelHitSizesY[5].Fill(nLayers[5,5], weight)
                        h_stripSelections[5].Fill(nLayers[5,6], weight)
                        h_hitPosXs[5].Fill(nLayers[5,7], weight)
                        h_hitPosYs[5].Fill(nLayers[5,8], weight)

                    if(plotEvent==True):
                        h_nPV.Fill(event.nPV, weight)

                print("Looking at dataset: " + dataset + ", tree: " +  tree + ", with " + str(trackCount) + "passing tracks.")
            out.cd() 
             
            print("Saving plots named " + track_names[trackNameCounter], trackNameCounter)
   
            # Write the histograms to root file           
            h_trackIso.Write('h_trackIso_' + track_names[trackNameCounter])
            h_eta.Write('h_eta_' + track_names[trackNameCounter])
            h_phi.Write('h_phi_' + track_names[trackNameCounter])
            h_nPV.Write('h_nPV_' + track_names[trackNameCounter])
            h_drMinJet.Write('h_drMinJet_' + track_names[trackNameCounter])
            h_ecalo.Write('h_ecalo_' + track_names[trackNameCounter])
            h_pt.Write('h_pt_' + track_names[trackNameCounter])
            h_d0.Write('h_d0_' + track_names[trackNameCounter])
            h_dz.Write('h_dz_' + track_names[trackNameCounter])
            h_charge.Write('h_charge_' + track_names[trackNameCounter])
            h_pixelHits.Write('h_pixelHits_' + track_names[trackNameCounter])
            h_Hits.Write('h_Hits_' + track_names[trackNameCounter])
            h_missingOuter.Write('h_missingOuter_' + track_names[trackNameCounter])
            h_dEdxPixel.Write('h_dEdxPixel_' + track_names[trackNameCounter])
            h_dEdxStrip.Write('h_dEdxStrip_' + track_names[trackNameCounter])
            h_pixelMeasurements.Write('h_pixelMeasurements_' + track_names[trackNameCounter])
            h_stripMeasurements.Write('h_stripMeasurements_' + track_names[trackNameCounter])
            h_pixelSat.Write('h_pixelSat_' + track_names[trackNameCounter])
            h_stripSat.Write('h_stripSat_' + track_names[trackNameCounter])
            h_sumEnergy.Write('h_sumEnergy_' + track_names[trackNameCounter])
            h_diffEnergy.Write('h_diffEnergy_' + track_names[trackNameCounter])
            h_dz1.Write('h_dz1_' + track_names[trackNameCounter])
            h_d01.Write('h_d01_' + track_names[trackNameCounter])
            h_dz2.Write('h_dz2_' + track_names[trackNameCounter])
            h_d02.Write('h_d02_' + track_names[trackNameCounter])
            h_dz3.Write('h_dz3_' + track_names[trackNameCounter])
            h_d03.Write('h_d03_' + track_names[trackNameCounter])
            h_layers[0].Write('h_layer1_' + track_names[trackNameCounter])
            h_charges[0].Write('h_charge1_' + track_names[trackNameCounter])
            h_subDets[0].Write('h_subDet1_' + track_names[trackNameCounter])
            h_pixelHitSizes[0].Write('h_pixelHitSize1_' + track_names[trackNameCounter])
            h_pixelHitSizesX[0].Write('h_pixelHitSizeX1_' + track_names[trackNameCounter])
            h_pixelHitSizesY[0].Write('h_pixelHitSizeY1_' + track_names[trackNameCounter])
            h_stripSelections[0].Write('h_stripSelection1_' + track_names[trackNameCounter])
            h_hitPosXs[0].Write('h_hitPosX1_' + track_names[trackNameCounter])
            h_hitPosYs[0].Write('h_hitPosY1_' + track_names[trackNameCounter])
            h_layers[1].Write('h_layer2_' + track_names[trackNameCounter])
            h_charges[1].Write('h_charge2_' + track_names[trackNameCounter])
            h_subDets[1].Write('h_subDet2_' + track_names[trackNameCounter])
            h_pixelHitSizes[1].Write('h_pixelHitSize2_' + track_names[trackNameCounter])
            h_pixelHitSizesX[1].Write('h_pixelHitSizeX2_' + track_names[trackNameCounter])
            h_pixelHitSizesY[1].Write('h_pixelHitSizeY2_' + track_names[trackNameCounter])
            h_stripSelections[1].Write('h_stripSelection2_' + track_names[trackNameCounter])
            h_hitPosXs[1].Write('h_hitPosX2_' + track_names[trackNameCounter])
            h_hitPosYs[1].Write('h_hitPosY2_' + track_names[trackNameCounter])
            h_layers[2].Write('h_layer3_' + track_names[trackNameCounter])
            h_charges[2].Write('h_charge3_' + track_names[trackNameCounter])
            h_subDets[2].Write('h_subDet3_' + track_names[trackNameCounter])
            h_pixelHitSizes[2].Write('h_pixelHitSize3_' + track_names[trackNameCounter])
            h_pixelHitSizesX[2].Write('h_pixelHitSizeX3_' + track_names[trackNameCounter])
            h_pixelHitSizesY[2].Write('h_pixelHitSizeY3_' + track_names[trackNameCounter])
            h_stripSelections[2].Write('h_stripSelection3_' + track_names[trackNameCounter])
            h_hitPosXs[2].Write('h_hitPosX3_' + track_names[trackNameCounter])
            h_hitPosYs[2].Write('h_hitPosY3_' + track_names[trackNameCounter])
            h_layers[3].Write('h_layer4_' + track_names[trackNameCounter])
            h_charges[3].Write('h_charge4_' + track_names[trackNameCounter])
            h_subDets[3].Write('h_subDet4_' + track_names[trackNameCounter])
            h_pixelHitSizes[3].Write('h_pixelHitSize4_' + track_names[trackNameCounter])
            h_pixelHitSizesX[3].Write('h_pixelHitSizeX4_' + track_names[trackNameCounter])
            h_pixelHitSizesY[3].Write('h_pixelHitSizeY4_' + track_names[trackNameCounter])
            h_stripSelections[3].Write('h_stripSelection4_' + track_names[trackNameCounter])
            h_hitPosXs[3].Write('h_hitPosX4_' + track_names[trackNameCounter])
            h_hitPosYs[3].Write('h_hitPosY4_' + track_names[trackNameCounter])
            if(layers != -1 and layers < 5): continue
            h_layers[4].Write('h_layer5_' + track_names[trackNameCounter])
            h_charges[4].Write('h_charge5_' + track_names[trackNameCounter])
            h_subDets[4].Write('h_subDet5_' + track_names[trackNameCounter])
            h_pixelHitSizes[4].Write('h_pixelHitSize5_' + track_names[trackNameCounter])
            h_pixelHitSizesX[4].Write('h_pixelHitSizeX5_' + track_names[trackNameCounter])
            h_pixelHitSizesY[4].Write('h_pixelHitSizeY5_' + track_names[trackNameCounter])
            h_stripSelections[4].Write('h_stripSelection5_' + track_names[trackNameCounter])
            h_hitPosXs[4].Write('h_hitPosX5_' + track_names[trackNameCounter])
            h_hitPosYs[4].Write('h_hitPosY5_' + track_names[trackNameCounter])
            if(layers != -1 and layers < 6): continue
            h_layers[5].Write('h_layer6_' + track_names[trackNameCounter])
            h_charges[5].Write('h_charge6_' + track_names[trackNameCounter])
            h_subDets[5].Write('h_subDet6_' + track_names[trackNameCounter])
            h_pixelHitSizes[5].Write('h_pixelHitSize6_' + track_names[trackNameCounter])
            h_pixelHitSizesX[5].Write('h_pixelHitSizeX6_' + track_names[trackNameCounter])
            h_pixelHitSizesY[5].Write('h_pixelHitSizeY6_' + track_names[trackNameCounter])
            h_stripSelections[5].Write('h_stripSelection6_' + track_names[trackNameCounter])
            h_hitPosXs[5].Write('h_hitPosX6_' + track_names[trackNameCounter])
            h_hitPosYs[5].Write('h_hitPosY6_' + track_names[trackNameCounter])

            #clear the histograms for the next set
            h_trackIso.Reset()
            h_eta.Reset()
            h_phi.Reset()
            h_nPV.Reset()
            h_drMinJet.Reset()
            h_ecalo.Reset()
            h_pt.Reset()
            h_d0.Reset()
            h_dz.Reset()
            h_charge.Reset()
            h_pixelHits.Reset()
            h_Hits.Reset()
            h_missingOuter.Reset()
            h_dEdxPixel.Reset()
            h_dEdxStrip.Reset()
            h_pixelMeasurements.Reset()
            h_stripMeasurements.Reset()
            h_pixelSat.Reset()
            h_stripSat.Reset()
            h_sumEnergy.Reset()
            h_diffEnergy.Reset()
            h_dz1.Reset()
            h_d01.Reset()
            h_dz2.Reset()
            h_d02.Reset()
            h_dz3.Reset()
            h_d03.Reset()
            h_layers[0].Reset()
            h_charges[0].Reset()
            h_subDets[0].Reset()
            h_pixelHitSizes[0].Reset()
            h_pixelHitSizesX[0].Reset()
            h_pixelHitSizesY[0].Reset()
            h_stripSelections[0].Reset()
            h_hitPosXs[0].Reset()
            h_hitPosYs[0].Reset()
            h_layers[1].Reset()
            h_charges[1].Reset()
            h_subDets[1].Reset()
            h_pixelHitSizes[1].Reset()
            h_pixelHitSizesX[1].Reset()
            h_pixelHitSizesY[1].Reset()
            h_stripSelections[1].Reset()
            h_hitPosXs[1].Reset()
            h_hitPosYs[1].Reset()
            h_layers[2].Reset()
            h_charges[2].Reset()
            h_subDets[2].Reset()
            h_pixelHitSizes[2].Reset()
            h_pixelHitSizesX[2].Reset()
            h_pixelHitSizesY[2].Reset()
            h_stripSelections[2].Reset()
            h_hitPosXs[2].Reset()
            h_hitPosYs[2].Reset()
            h_layers[3].Reset()
            h_charges[3].Reset()
            h_subDets[3].Reset()
            h_pixelHitSizes[3].Reset()
            h_pixelHitSizesX[3].Reset()
            h_pixelHitSizesY[3].Reset()
            h_stripSelections[3].Reset()
            h_hitPosXs[3].Reset()
            h_hitPosYs[3].Reset()
            if(layers != -1 and layers < 5): continue
            h_layers[4].Reset()
            h_charges[4].Reset()
            h_subDets[4].Reset()
            h_pixelHitSizes[4].Reset()
            h_pixelHitSizesX[4].Reset()
            h_pixelHitSizesY[4].Reset()
            h_stripSelections[4].Reset()
            h_hitPosXs[4].Reset()
            h_hitPosYs[4].Reset()
            if(layers != -1 and layers < 6): continue
            h_layers[5].Reset()
            h_charges[5].Reset()
            h_subDets[5].Reset()
            h_pixelHitSizes[5].Reset()
            h_pixelHitSizesX[5].Reset()
            h_pixelHitSizesY[5].Reset()
            h_stripSelections[5].Reset()
            h_hitPosXs[5].Reset()
            h_hitPosYs[5].Reset()

            trackNameCounter += 1
            print("Increased track name counter " + str(trackNameCounter))

    out.Close()
            


def plotTogether(base_name, dataset_names, dataType, filename, outfile = 'combinedPlots_4PlusLayer_v2Test.root'):

    colors = [1, 807, 920, 4, 6, 7, 9, 2, 3]
    outfile = r.TFile(outfile, 'update')
    file = r.TFile.Open(filename)
    histograms = []
    lineStyle = []

    for i in range(len(dataset_names)):
        h = file.Get(base_name+dataset_names[i])
        print(base_name+dataset_names[i])
        if(h.Integral() > 0): 
            print("Histogram names", dataset_names[i], h.GetName())
            h.Sumw2()
            h.Scale(float(1)/h.Integral())
            h.SetName('h' + str(i))
        histograms.append(h)
        if "pileup" in dataset_names[i]: lineStyle.append(8)
        elif "fake" in dataset_names[i]: lineStyle.append(2) 
        elif "real" in dataset_names[i]: lineStyle.append(1)
        else: print("This histogram is not pileup, fake, or real", dataset_names[i])
            

    l1 = r.TLegend(0.7, 0.7, 0.8, 0.9)
    c1 = r.TCanvas("c1", base_name, 800, 800)
    c1.cd()
    for hist in range(len(histograms)):
        histograms[hist].SetMaximum(1)
        if hist == 0:
            if(histograms[hist].GetEntries()>0): 
                histograms[hist].Draw("HIST")
                histograms[hist].SetLineColor(colors[hist])
                l1.AddEntry(histograms[hist], dataset_names[hist], "l")
        else:
            if(histograms[hist].GetEntries()>0): 
                histograms[hist].Draw("HIST SAMES")
                #histograms[hist].SetLineStyle(lineStyle[hist])
                histograms[hist].SetLineColor(colors[hist])
                #if('fake' in dataset_names[hist]): histograms[hist].SetLineStyle(2)
                l1.AddEntry(histograms[hist], dataset_names[hist], "l")


    l1.Draw("SAME")
    r.gStyle.SetOptStat(0)
    outfile.cd()
    c1.Write(base_name)

    outfile.Close()
    file.Close()
    c1.Close()
    c1.Delete()
    del histograms



if __name__ == "__main__":

    trees = ['realTree', 'fakeTree', 'pileupTree']
    zerobias_track_names = ['real_zb', 'fake_zb', 'pileup_zb']
    dy_track_names = ['real_dy', 'fake_dy', 'pileup_dy']
    neutrino_names = ['real_nu', 'fake_nu', 'pileup_nu']
    higgsino_10_names = ['real_10', 'fake_10']
    higgsino_100_names = ['real_100', 'fake_100']
    higgsino_1000_names = ['real_1000', 'fake_1000']
    higgsino_10000_names = ['real_10000', 'fake_10000']

    track_names = zerobias_track_names + neutrino_names + dy_track_names
    datasets = zeroBiasData + neutrinoGunData + dyData
    dataType = ['real', 'mc', 'mc']

    scale_factors_dy = makePUScaleFactors('puData_2017D_central.root', '/share/scratch0/mcarrigan/disTracksML/CMSSW_9_4_9/src/DisappTrks/StandardAnalysis/data/pu_disappTrks_run2.root')
    scale_factors_NG = makePUScaleFactors('puData_2017D_central.root', '/share/scratch0/mcarrigan/disTracksML/CMSSW_9_4_9/src/DisappTrks/StandardAnalysis/data/pu_disappTrks_run2.root')

    PU_dict = {dyData[0]:scale_factors_dy, neutrinoGunData[0]:scale_factors_NG}

    plotDatasets(datasets, trees, track_names, dataType)

    out.Close()

    plottedVariableFile = "plottedVariables.root"

    plotTogether('h_trackIso_', track_names,dataType, plottedVariableFile)
    plotTogether('h_eta_', track_names,dataType, plottedVariableFile)
    plotTogether('h_phi_', track_names,dataType, plottedVariableFile)
    plotTogether('h_nPV_', track_names,dataType, plottedVariableFile)
    plotTogether('h_drMinJet_', track_names,dataType, plottedVariableFile)
    plotTogether('h_ecalo_', track_names,dataType, plottedVariableFile)
    plotTogether('h_d0_', track_names,dataType, plottedVariableFile)
    plotTogether('h_dz_', track_names,dataType, plottedVariableFile)
    plotTogether('h_pt_', track_names,dataType, plottedVariableFile)
    plotTogether('h_charge_', track_names,dataType, plottedVariableFile)
    plotTogether('h_pixelHits_', track_names,dataType, plottedVariableFile)
    plotTogether('h_Hits_', track_names,dataType, plottedVariableFile)
    plotTogether('h_missingOuter_', track_names,dataType, plottedVariableFile)
    plotTogether('h_dEdxPixel_', track_names,dataType, plottedVariableFile)
    plotTogether('h_pixelMeasurements_', track_names,dataType, plottedVariableFile)
    plotTogether('h_stripMeasurements_', track_names,dataType, plottedVariableFile)
    plotTogether('h_pixelSat_', track_names,dataType, plottedVariableFile)
    plotTogether('h_stripSat_', track_names,dataType, plottedVariableFile)
    plotTogether('h_sumEnergy_', track_names,dataType, plottedVariableFile)
    plotTogether('h_diffEnergy_', track_names,dataType, plottedVariableFile)
    plotTogether('h_dz1_', track_names,dataType, plottedVariableFile)
    plotTogether('h_d01_', track_names,dataType, plottedVariableFile)
    plotTogether('h_dz2_', track_names,dataType, plottedVariableFile)
    plotTogether('h_d02_', track_names,dataType, plottedVariableFile)
    plotTogether('h_dz3_', track_names,dataType, plottedVariableFile)
    plotTogether('h_d03_', track_names,dataType, plottedVariableFile)
    plotTogether('h_layer1_', track_names,dataType, plottedVariableFile)
    plotTogether('h_charge1_', track_names,dataType, plottedVariableFile)
    plotTogether('h_subDet1_', track_names,dataType, plottedVariableFile)
    plotTogether('h_pixelHitSize1_', track_names,dataType, plottedVariableFile)
    plotTogether('h_pixelHitSizeX1_', track_names,dataType, plottedVariableFile)
    plotTogether('h_pixelHitSizeY1_', track_names,dataType, plottedVariableFile)
    plotTogether('h_stripSelection1_', track_names,dataType, plottedVariableFile)
    plotTogether('h_hitPosX1_', track_names,dataType, plottedVariableFile)
    plotTogether('h_hitPosY1_', track_names,dataType, plottedVariableFile)
    plotTogether('h_layer2_', track_names,dataType, plottedVariableFile)
    plotTogether('h_charge2_', track_names,dataType, plottedVariableFile)
    plotTogether('h_subDet2_', track_names,dataType, plottedVariableFile)
    plotTogether('h_pixelHitSize2_', track_names,dataType, plottedVariableFile)
    plotTogether('h_pixelHitSizeX2_', track_names,dataType, plottedVariableFile)
    plotTogether('h_pixelHitSizeY2_', track_names,dataType, plottedVariableFile)
    plotTogether('h_stripSelection2_', track_names,dataType, plottedVariableFile)
    plotTogether('h_hitPosX2_', track_names,dataType, plottedVariableFile)
    plotTogether('h_hitPosY2_', track_names,dataType, plottedVariableFile)
    plotTogether('h_layer3_', track_names,dataType, plottedVariableFile)
    plotTogether('h_charge3_', track_names,dataType, plottedVariableFile)
    plotTogether('h_subDet3_', track_names,dataType, plottedVariableFile)
    plotTogether('h_pixelHitSize3_', track_names,dataType, plottedVariableFile)
    plotTogether('h_pixelHitSizeX3_', track_names,dataType, plottedVariableFile)
    plotTogether('h_pixelHitSizeY3_', track_names,dataType, plottedVariableFile)
    plotTogether('h_stripSelection3_', track_names,dataType, plottedVariableFile)
    plotTogether('h_hitPosX3_', track_names,dataType, plottedVariableFile)
    plotTogether('h_hitPosY3_', track_names,dataType, plottedVariableFile)
    plotTogether('h_layer4_', track_names,dataType, plottedVariableFile)
    plotTogether('h_charge4_', track_names,dataType, plottedVariableFile)
    plotTogether('h_subDet4_', track_names,dataType, plottedVariableFile)
    plotTogether('h_pixelHitSize4_', track_names,dataType, plottedVariableFile)
    plotTogether('h_pixelHitSizeX4_', track_names,dataType, plottedVariableFile)
    plotTogether('h_pixelHitSizeY4_', track_names,dataType, plottedVariableFile)
    plotTogether('h_stripSelection4_', track_names,dataType, plottedVariableFile)
    plotTogether('h_hitPosX4_', track_names,dataType, plottedVariableFile)
    plotTogether('h_hitPosY4_', track_names,dataType, plottedVariableFile)
    plotTogether('h_layer5_', track_names,dataType, plottedVariableFile)
    plotTogether('h_charge5_', track_names,dataType, plottedVariableFile)
    plotTogether('h_subDet5_', track_names,dataType, plottedVariableFile)
    plotTogether('h_pixelHitSize5_', track_names,dataType, plottedVariableFile)
    plotTogether('h_pixelHitSizeX5_', track_names,dataType, plottedVariableFile)
    plotTogether('h_pixelHitSizeY5_', track_names,dataType, plottedVariableFile)
    plotTogether('h_stripSelection5_', track_names,dataType, plottedVariableFile)
    plotTogether('h_hitPosX5_', track_names,dataType, plottedVariableFile)
    plotTogether('h_hitPosY5_', track_names,dataType, plottedVariableFile)
    plotTogether('h_layer6_', track_names,dataType, plottedVariableFile)
    plotTogether('h_charge6_', track_names,dataType, plottedVariableFile)
    plotTogether('h_subDet6_', track_names,dataType, plottedVariableFile)
    plotTogether('h_pixelHitSize6_', track_names,dataType, plottedVariableFile)
    plotTogether('h_pixelHitSizeX6_', track_names,dataType, plottedVariableFile)
    plotTogether('h_pixelHitSizeY6_', track_names,dataType, plottedVariableFile)
    plotTogether('h_stripSelection6_', track_names,dataType, plottedVariableFile)
    plotTogether('h_hitPosX6_', track_names,dataType, plottedVariableFile)
    plotTogether('h_hitPosY6_', track_names,dataType, plottedVariableFile)






















