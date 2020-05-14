import FWCore.ParameterSet.Config as cms
import glob, sys, os

# The following are needed for the calculation of associated calorimeter energy
from Configuration.StandardSequences.GeometryRecoDB_cff import *
from Configuration.StandardSequences.MagneticField_38T_cff import *

###########################################################
##### Set up process #####
###########################################################

process = cms.Process ('IMAGES')
process.load ('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.cerr.FwkReport.reportEvery = 1

process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '94X_mc2017_realistic_v11', '')

process.maxEvents = cms.untracked.PSet (
    input = cms.untracked.int32 (100)
)
process.source = cms.Source ("PoolSource",
    fileNames = cms.untracked.vstring (
    	'/store/mc/RunIIFall17DRPremix/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/AODSIM/RECOSIMstep_94X_mc2017_realistic_v10_ext1-v1/00001/64649185-CAF3-E711-955D-02163E014685.root',
        #'/store/mc/RunIIFall17DRPremix/DYJetsToLL_M-5to50_TuneCP5_13TeV-madgraphMLM-pythia8/AODSIM/PU2017_94X_mc2017_realistic_v11-v3/280000/0002944B-672B-E911-8E3C-0025905B85CC.root',
    ),
)
process.TFileService = cms.Service ('TFileService',
    fileName = cms.string ('images.root')
)

###########################################################
##### Set up the producer and the end path            #####
###########################################################

process.trackImageAnalyzer = cms.EDAnalyzer ("TrackImageProducer",
    tracks       = cms.InputTag   ("generalTracks"),
    genParticles = cms.InputTag   ("genParticles", ""),
    electrons    = cms.InputTag   ("gedGsfElectrons", ""),
    muons        = cms.InputTag   ("muons", ""),
    taus         = cms.InputTag   ("hpsPFTauProducer", ""),
    pfCandidates = cms.InputTag   ("particleFlow", ""),
    EBRecHits    =  cms.InputTag  ("reducedEcalRecHitsEB"),
    EERecHits    =  cms.InputTag  ("reducedEcalRecHitsEE"),
    HBHERecHits  =  cms.InputTag  ("reducedHcalRecHits", "hbhereco"),

    tauDecayModeFinding      = cms.InputTag ("hpsPFTauDiscriminationByDecayModeFinding"),
    tauElectronDiscriminator = cms.InputTag ("hpsPFTauDiscriminationByMVA6LooseElectronRejection"),
    tauMuonDiscriminator     = cms.InputTag ("hpsPFTauDiscriminationByLooseMuonRejection3"),

    minGenParticlePt = cms.double (10.0),
    minTrackPt       = cms.double (35.0),
    maxRelTrackIso   = cms.double (0.05),

    maxDEtaTrackRecHit = cms.double (0.5),
    maxDPhiTrackRecHit = cms.double (0.5),
)

process.myPath = cms.Path (process.trackImageAnalyzer)