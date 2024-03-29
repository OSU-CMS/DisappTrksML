#!/usr/bin/env python
import os
import sys
import math
import ctypes
import glob

from ROOT import TChain, TFile, TTree, TH1D, TH2D, TCanvas, THStack, gROOT, gStyle, TLegend, TGraph, TMarker

from DisappTrksML.DeepSets.ElectronModel import *
from DisappTrksML.DeepSets.MuonModel import *

def calculateFidicualMaps(electron_payload, muon_payload, payload_suffix):
    fiducial_maps = {}

    for flavor in ['ele', 'muon']:
        fin = TFile(electron_payload if flavor is 'ele' else muon_payload, 'read')
        beforeVeto = fin.Get('beforeVeto' + payload_suffix)
        afterVeto = fin.Get('afterVeto' + payload_suffix)

        totalEventsBeforeVeto = 0
        totalEventsAfterVeto = 0
        nRegionsWithTag = 0

        # loop over all bins in eta-phi and count events before/after to calculate the mean inefficiency
        for xbin in range(1, beforeVeto.GetXaxis().GetNbins() + 1):
            for ybin in range(1, beforeVeto.GetYaxis().GetNbins() + 1):
                if beforeVeto.GetBinContent(xbin, ybin) > 0:
                    nRegionsWithTag += 1
                else:
                    continue
                totalEventsBeforeVeto += beforeVeto.GetBinContent(xbin, ybin)
                totalEventsAfterVeto  += afterVeto.GetBinContent(xbin, ybin)
        meanInefficiency = totalEventsAfterVeto / totalEventsBeforeVeto

        # now with the mean, calculate the standard deviation as stdDev^2 = 1/(N-1) * sum(inefficiency - meanInefficiency)^2
        stdDevInefficiency = 0

        efficiency = afterVeto.Clone(flavor + '_efficiency')
        efficiency.SetDirectory(0)
        efficiency.Divide(beforeVeto)

        for xbin in range(1, efficiency.GetXaxis().GetNbins() + 1):
            for ybin in range(1, efficiency.GetYaxis().GetNbins() + 1):
                if beforeVeto.GetBinContent(xbin, ybin) == 0:
                    continue
                thisInefficiency = efficiency.GetBinContent(xbin, ybin)
                stdDevInefficiency += (thisInefficiency - meanInefficiency)**2

        if nRegionsWithTag < 2:
            print('Only ', nRegionsWithTag, ' regions with a tag lepton exist, cannot calculate fiducial map!!!')
            return hotSpots

        stdDevInefficiency /= nRegionsWithTag - 1
        stdDevInefficiency = math.sqrt(stdDevInefficiency)

        for xbin in range(1, efficiency.GetXaxis().GetNbins() + 1):
            for ybin in range(1, efficiency.GetYaxis().GetNbins() + 1):
                if beforeVeto.GetBinContent(xbin, ybin) == 0:
                    continue
                efficiency.SetBinContent(xbin, ybin, efficiency.GetBinContent(xbin, ybin) - meanInefficiency)
        efficiency.Scale(1. / stdDevInefficiency)

        fiducial_maps[flavor] = efficiency

    return fiducial_maps

def fiducialMapSigma(track, fiducial_maps):
    maxSigma = -999.

    iBin = fiducial_maps['ele'].GetXaxis().FindBin(track.eta)
    jBin = fiducial_maps['ele'].GetYaxis().FindBin(track.phi)

    for i in range(iBin-1, iBin+2):
        if i > fiducial_maps['ele'].GetNbinsX() or i < 0: continue
        for j in range(jBin-1, jBin+2):
            if j > fiducial_maps['ele'].GetNbinsY() or j < 0: continue

            dEta = track.eta - fiducial_maps['ele'].GetXaxis().GetBinCenter(i)
            dPhi = track.phi - fiducial_maps['ele'].GetYaxis().GetBinCenter(j)

            if dEta*dEta + dPhi*dPhi > 0.05*0.5:
                continue
            
            if fiducial_maps['ele'].GetBinContent(i, j) > maxSigma:
                maxSigma = fiducial_maps['ele'].GetBinContent(i, j)

            if fiducial_maps['muon'].GetBinContent(i, j) > maxSigma:
                maxSigma = fiducial_maps['muon'].GetBinContent(i, j)

    return maxSigma

def processFile(fileName, models, deep_sets_arch, fiducial_maps, dataset):
    # output: np.array[iEvent] = [discriminant[m] for m in models] + [nLayersWithMeasurement, sigma]
    values = []

    print("Processing file: " + fileName)

    inputFile = TFile(fileName, 'read')
    tree = inputFile.Get('trackImageProducer/tree')

    for iEvent, event in enumerate(tree):
        if dataset.startswith('higgsino'):
            eventPasses, trackPasses = deep_sets_arch.eventSelectionSignal(event)
        elif dataset == 'fake':
            eventPasses, trackPasses = deep_sets_arch.eventSelectionFakeBackground(event)
        else:
            eventPasses, trackPasses, trackPassesVeto = deep_sets_arch.eventSelectionLeptonBackground(event, dataset)

        if not eventPasses: continue

        for iTrack, track in enumerate(event.tracks):
            if not trackPasses[iTrack]: continue

            track_values = [-1] * (len(models) + 2)

            sigma = fiducialMapSigma(track, fiducial_maps)

            track_values[-2] = track.nLayersWithMeasurement
            track_values[-1] = sigma

            if dataset in ['electrons', 'muons'] and not trackPassesVeto[iTrack]:
                values.append(track_values)
                continue

            if(len(models)>0): track_values[0] = models['ele_14'].evaluate_model(event, track)
            if(len(models)>1): track_values[1] = models['ele_17'].evaluate_model(event, track)
            if(len(models)>2): track_values[2] = models['ele_19'].evaluate_model(event, track)
            if(len(models)>3): track_values[3] = models['muon'].evaluate_model(event, track)

            values.append(track_values)

    inputFile.Close()

    return len(values), np.array(values)

def processDataset(dataset, inputDirs, models, deep_sets_arch, fiducial_maps):
    if os.path.exists('output_' + dataset + '.root'):
        return

    # output: np.array[iEvent] = [discriminant[m] for m in models] + [nLayersWithMeasurement, sigma]
    results = np.empty((0, len(models) + 2), float)

    inputFiles = []
    for inputDir in inputDirs:
    	inputFiles.extend(glob.glob(inputDir))

    nFiles = len(inputFiles)

    for iFile, inputFile in enumerate(inputFiles):
    	if iFile % 10 == 0:
    		print('\tFile', iFile, '/', nFiles, '...')

        #FIXME: just for debugging
    	if iFile > 2: break

    	n, new_results = processFile(inputFile, models, deep_sets_arch, fiducial_maps, dataset)
    	if n > 0:
    		results = np.append(results, new_results, axis=0)

    np.savez_compressed('outputFiles/output_' + dataset + '.npz', results=results)

######################################

def getROC(h_ele, h_mu, h_fake, h_signal, nlayers):
    ibin = h_ele.GetYaxis().FindBin(nlayers)

    h1d_bkg = h_ele.ProjectionX('roc1d_ele_'      + str(nlayers), ibin, ibin if nlayers < 6 else -1)
    h1d_bkg.Add(h_mu.ProjectionX('roc1d_mu_'      + str(nlayers), ibin, ibin if nlayers < 6 else -1))
    h1d_bkg.Add(h_fake.ProjectionX('roc1d_fake_'  + str(nlayers), ibin, ibin if nlayers < 6 else -1))
    h1d_sig  = h_signal.ProjectionX('roc1d_sig_'  + str(nlayers), ibin, ibin if nlayers < 6 else -1)

    n_bkg = h1d_bkg.Integral()
    n_sig = h1d_sig.Integral()

    graph = TGraph()

    best_sOverRootB = 1
    best_cut = h1d_bkg.GetBinLowEdge(h1d_bkg.GetNbinsX() + 1)
    best_cut_x = best_cut_y = 1

    i = 0
    for j in reversed(list(range(h1d_bkg.GetNbinsX()))):
        n_pass_bkg = h1d_bkg.Integral(1, j+1)
        n_pass_sig = h1d_sig.Integral(1, j+1)

        graph.SetPoint(i, n_pass_bkg / n_bkg if n_bkg > 0 else 0, n_pass_sig / n_sig if n_sig > 0 else 0)
        i += 1

        if n_pass_bkg > 0 and n_sig > 0 and n_bkg > 0:
            sOverRootB = n_pass_sig / n_sig / math.sqrt(n_pass_bkg / n_bkg)
            if sOverRootB > best_sOverRootB:
                best_sOverRootB = sOverRootB
                best_cut = h1d_bkg.GetBinLowEdge(j+2)
                best_cut_x = n_pass_bkg / n_bkg
                best_cut_y = n_pass_sig / n_sig

    return (graph, best_cut, best_cut_x, best_cut_y)

def getWP(h_ele, h_ele_sigma, h_mu, h_mu_sigma, h_fake, h_fake_sigma, h_signal, h_signal_sigma, nlayers):
    ibin = h_ele.GetYaxis().FindBin(nlayers)

    bkg_ele  = h_ele.ProjectionX   ('ele_'  + str(nlayers), ibin, ibin if nlayers < 6 else -1)
    bkg_muon = h_mu.ProjectionX  ('muon_' + str(nlayers), ibin, ibin if nlayers < 6 else -1)
    bkg_fake = h_fake.ProjectionX  ('fake_' + str(nlayers), ibin, ibin if nlayers < 6 else -1)
    signal   = h_signal.ProjectionX('sig_'  + str(nlayers), ibin, ibin if nlayers < 6 else -1)

    bkg_ele_sigma  = h_ele_sigma.ProjectionX   ('ele_'  + str(nlayers), ibin, ibin if nlayers < 6 else -1)
    bkg_muon_sigma = h_mu_sigma.ProjectionX  ('muon_' + str(nlayers), ibin, ibin if nlayers < 6 else -1)
    bkg_fake_sigma = h_fake_sigma.ProjectionX  ('fake_' + str(nlayers), ibin, ibin if nlayers < 6 else -1)
    signal_sigma   = h_signal_sigma.ProjectionX('sig_'  + str(nlayers), ibin, ibin if nlayers < 6 else -1)

    ibin = bkg_ele.FindBin(0.2)
    n_pass_bkg = bkg_ele.Integral(1, ibin) + bkg_muon.Integral(1, ibin) + bkg_fake.Integral(1, ibin)
    n_pass_signal = signal.Integral(1, ibin)

    ibin = bkg_ele_sigma.FindBin(2.0)
    n_pass_bkg_sigma = bkg_ele_sigma.Integral(0, ibin) + bkg_muon_sigma.Integral(0, ibin) + bkg_fake_sigma.Integral(0, ibin)
    n_pass_signal_sigma = signal_sigma.Integral(0, ibin)

    change_bkg = n_pass_bkg / n_pass_bkg_sigma if n_pass_bkg_sigma > 0 else 1.0
    change_signal = n_pass_signal / n_pass_signal_sigma if n_pass_signal_sigma > 0 else 1.0

    return (change_bkg, change_signal)

def plotStack(h_ele, h_muon, h_fake, h_signal, nlayers, output_prefix, canvas, rebin=-1):
    ibin = h_ele.GetYaxis().FindBin(nlayers)

    bkg_ele  = h_ele.ProjectionX   ('ele_'  + str(nlayers), ibin, ibin if nlayers < 6 else -1)
    bkg_muon = h_muon.ProjectionX  ('muon_' + str(nlayers), ibin, ibin if nlayers < 6 else -1)
    bkg_fake = h_fake.ProjectionX  ('fake_' + str(nlayers), ibin, ibin if nlayers < 6 else -1)
    signal   = h_signal.ProjectionX('sig_'  + str(nlayers), ibin, ibin if nlayers < 6 else -1)

    bkg_ele.SetFillStyle(1001)
    bkg_ele.SetFillColor(400 - 7) # kYellow-7
    bkg_ele.SetLineColor(1)
    bkg_ele.Add(bkg_muon)
    bkg_ele.Add(bkg_fake)

    bkg_muon.SetFillStyle(1001)
    bkg_muon.SetFillColor(632) # kRed
    bkg_muon.SetLineColor(1)
    bkg_muon.Add(bkg_fake)

    bkg_fake.SetFillStyle(1001)
    bkg_fake.SetFillColor(8) # ~green
    bkg_fake.SetLineColor(1)
        
    signal.SetLineColor(1)
    signal.SetLineWidth(3)

    if rebin > 0:
        bkg_ele.Rebin(rebin)
        bkg_muon.Rebin(rebin)
        bkg_fake.Rebin(rebin)
        signal.Rebin(rebin)

    bkg_ele.GetXaxis().SetTitle('Discriminant')
    bkg_ele.GetYaxis().SetTitle('Events')

    bkg_ele.Draw('hist')
    bkg_muon.Draw('hist same')
    bkg_fake.Draw('hist same')
    signal.Draw('same')

    legend = TLegend(0.55, 0.7, 0.85, 0.85)
    legend.SetBorderSize(0)
    legend.SetFillColor(0)
    legend.SetFillStyle(0)
    legend.SetTextFont(42)

    legend.AddEntry(bkg_ele, 'Electrons', 'F')
    legend.AddEntry(bkg_muon, 'Muons', 'F')
    legend.AddEntry(bkg_fake, 'Fakes', 'F')
    legend.AddEntry(signal, 'Higgsino 300_1000', 'L')
    legend.Draw('same')

    canvas.SaveAs(output_prefix + '_' + str(nlayers) + '.pdf')

def printChanges(label, h_sigma, h_disc, disc_cut, nlayers, iClone):
    ibin = h_sigma.GetYaxis().FindBin(nlayers)
    iClone += 1

    h_sigma_proj = h_sigma.ProjectionX('sigma_proj_' + str(iClone), ibin, ibin if nlayers < 6 else -1)
    h_disc_proj  = h_disc.ProjectionX( 'disc_proj_' + str(iClone), ibin, ibin if nlayers < 6 else -1)

    n_pass_sigma = h_sigma_proj.Integral(0, h_sigma_proj.FindBin(2.0))
    n_pass_disc = h_disc_proj.Integral(0, h_disc_proj.FindBin(disc_cut))

    eff_sigma = n_pass_sigma / h_sigma_proj.Integral() if h_sigma_proj.Integral() > 0 else 0
    eff_disc = n_pass_disc / h_disc_proj.Integral() if h_disc_proj.Integral() > 0 else 0

    eff_ratio = eff_disc / eff_sigma if eff_sigma > 0 else 0

    print(label, eff_ratio)

    return label + ' ' + str(eff_ratio)

def analyze(datasets, inputDir='.'):
    gROOT.SetBatch()
    gStyle.SetOptStat(0)
    gStyle.SetOptTitle(0)

    canvas = TCanvas("c1", "c1", 800, 800)

    '''
    input_ele = TFile(inputDir + '/output_electrons.root', 'read')
    input_muon = TFile(inputDir + '/output_muons.root', 'read')
    input_fake = TFile(inputDir + '/output_fake.root', 'read')
    '''

    input_ele = np.load(inputDir + '/output_electrons.npz')['results']
    input_muon = np.load(inputDir + '/output_muons.npz')['results']
    input_fake = np.load(inputDir + '/output_fake.npz')['results']

    h_disc_ele  = TH2D('disc_ele', 'disc:discriminant:nLayers', 3000, 0, 1, 3, 4, 7)
    h_sigma_ele = TH2D('sigma_ele', 'sigma:sigma:nLayers', 3000, 0, 30, 3, 4, 7)

    h_disc_muon  = TH2D('disc_muon', 'disc:discriminant:nLayers', 3000, 0, 1, 3, 4, 7)
    h_sigma_muon = TH2D('sigma_muon', 'sigma:sigma:nLayers', 3000, 0, 30, 3, 4, 7)

    h_disc_fake  = TH2D('disc_fake', 'disc:discriminant:nLayers', 3000, 0, 1, 3, 4, 7)
    h_sigma_fake = TH2D('sigma_fake', 'sigma:sigma:nLayers', 3000, 0, 30, 3, 4, 7)

    for disc in input_ele:
        h_disc_ele.Fill(disc[0], disc[-2]) #note that second to last index is the nLayers
        h_sigma_ele.Fill(disc[-1], disc[-2]) #last index is currently sigma

    for disc in input_muon:
        h_disc_muon.Fill(disc[0], disc[-2]) #note that second to last index is the nLayers
        h_sigma_muon.Fill(disc[-1], disc[-2]) #last index is currently sigma

    for disc in input_fake:
        h_disc_fake.Fill(disc[0], disc[-2]) #note that second to last index is the nLayers
        h_sigma_fake.Fill(disc[-1], disc[-2]) #last index is currently sigma

    '''
    h_disc_ele = input_ele.Get('disc')
    h_disc_muon = input_muon.Get('disc')
    h_disc_fake = input_fake.Get('disc')

    h_sigma_ele = input_ele.Get('sigma')
    h_sigma_muon = input_muon.Get('sigma')
    h_sigma_fake = input_fake.Get('sigma')
    '''

    h_disc_signal = {}
    h_sigma_signal = {}
    for dataset in datasets:
        print(dataset)
        if not dataset.startswith('higgsino'): continue
        input_signal = np.load(inputDir + '/output_' + dataset + '.npz')['results']
     
        '''
        fin = TFile(inputDir + '/output_' + dataset + '.root', 'read')

        h_disc = fin.Get('disc')
        h_sigma = fin.Get('sigma')

        h_disc.SetDirectory(0)
        h_sigma.SetDirectory(0)
        '''

        h_disc = TH2D('disc_signal', 'disc:discriminant:nLayers', 3000, 0, 1, 3, 4, 7)
        h_sigma = TH2D('sigma_signal', 'sigma:sigma:nLayers', 3000, 0, 30, 3, 4, 7)

        for disc in input_signal:
            h_disc.Fill(disc[0], disc[-2])
            h_sigma.Fill(disc[-1], disc[-2])

        h_disc_signal[dataset] = h_disc
        h_sigma_signal[dataset] = h_sigma

        #fin.Close()

    roc_curves = { 'disc_%d_%s' % (i, sig) : getROC(h_disc_ele, h_disc_muon, h_disc_fake, h_disc_signal[sig], i) for i in [4, 5, 6] for sig in h_disc_signal }
    roc_curves.update(
        { 'sigma_%d_%s' % (i, sig) : getROC(h_sigma_ele, h_sigma_muon, h_sigma_fake, h_sigma_signal[sig], i) for i in [4, 5, 6] for sig in h_disc_signal }
    )

    fout = TFile('durp.root', 'recreate')
    for curve_name in roc_curves:
        roc_curves[curve_name][0].Write(curve_name)
        print('Optimal cut for', curve_name, '=', roc_curves[curve_name][1])
        m = TMarker(roc_curves[curve_name][2], roc_curves[curve_name][3], 29)
        m.Write(curve_name + '_optimal')
    fout.Close()

    for i in [4, 5, 6]:
        plotStack(h_disc_ele, h_disc_muon, h_disc_fake, h_disc_signal['higgsino_700_1000'], i, 'discriminant', canvas, rebin=5)
        plotStack(h_sigma_ele, h_sigma_muon, h_sigma_fake, h_sigma_signal['higgsino_700_1000'], i, 'sigma', canvas, rebin=5)

    for dataset in h_disc_signal:
        for i in [4, 5, 6]:
            wp = getWP(h_disc_ele, h_sigma_ele, h_disc_muon, h_sigma_muon, h_disc_fake, h_sigma_fake, h_disc_signal[dataset], h_sigma_signal[dataset], i)
            print(dataset, '--', wp)

    iClone = -1

    print('\nComparing disc < 0.3 to fiducial map method:\n')
    printChanges('Electrons nLayers=4',  h_sigma_ele, h_disc_ele, 0.3, 4, iClone)
    printChanges('Electrons nLayers=5',  h_sigma_ele, h_disc_ele, 0.3, 5, iClone)
    printChanges('Electrons nLayers>=6', h_sigma_ele, h_disc_ele, 0.3, 6, iClone)

    printChanges('Muons nLayers=4',  h_sigma_muon, h_disc_muon, 0.3, 4, iClone)
    printChanges('Muons nLayers=5',  h_sigma_muon, h_disc_muon, 0.3, 5, iClone)
    printChanges('Muons nLayers>=6', h_sigma_muon, h_disc_muon, 0.3, 6, iClone)

    printChanges('Fakes nLayers=4',  h_sigma_fake, h_disc_fake, 0.3, 4, iClone)
    printChanges('Fakes nLayers=5',  h_sigma_fake, h_disc_fake, 0.3, 5, iClone)
    printChanges('Fakes nLayers>=6', h_sigma_fake, h_disc_fake, 0.3, 6, iClone)

    extra_samples = {
        10    : ['0p' + str(i) for i in range(2, 10)] + [str(i) for i in range(0, 10)],
        100   : [str(i*10) for i in range(2, 10)],
        1000  : [str(i*100) for i in range(2, 10)],
        10000 : [str(i*1000) for i in range(2, 10)],
    }

    for i in [4, 5, 6]:
        print('NLAYERS: ', i)
        for mass in range(100, 1000, 100):
            if mass != 700: continue # temporary because we only used 700
            for lifetime in [10, 100, 1000, 10000]:
                datasetName = 'higgsino_%d_%d' % (mass, lifetime)
                change_string = printChanges('Higgsino_%dGeV_%dcm_94X' % (mass, lifetime), 
                                             h_sigma_signal[datasetName], 
                                             h_disc_signal[datasetName], 
                                             0.3, 
                                             i,
                                             iClone)
                for extra_lifetime in extra_samples[lifetime]:
                    print(change_string.replace('Higgsino_%dGeV_%dcm_94X' % (mass, lifetime), 'Higgsino_%dGeV_%scm_94X' % (mass, extra_lifetime)))
