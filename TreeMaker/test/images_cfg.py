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
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

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
        # M>50
        '/store/mc/RunIIFall17DRPremix/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/AODSIM/RECOSIMstep_94X_mc2017_realistic_v10_ext1-v1/80001/8AD8A719-03FE-E711-BBE9-001E677925A0.root',
        '/store/mc/RunIIFall17DRPremix/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/AODSIM/RECOSIMstep_94X_mc2017_realistic_v10_ext1-v1/80001/E89A4682-03FE-E711-A8BF-001E67E6F990.root',
        '/store/mc/RunIIFall17DRPremix/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/AODSIM/RECOSIMstep_94X_mc2017_realistic_v10_ext1-v1/80001/CE2DD2B6-ABFE-E711-B66A-0242AC130002.root',
        '/store/mc/RunIIFall17DRPremix/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/AODSIM/RECOSIMstep_94X_mc2017_realistic_v10_ext1-v1/80001/4ED9C9CC-DBFE-E711-A64A-0242AC130002.root',
        '/store/mc/RunIIFall17DRPremix/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/AODSIM/RECOSIMstep_94X_mc2017_realistic_v10_ext1-v1/80001/E04706CD-8AFE-E711-85C1-0CC47A6C1054.root',
        '/store/mc/RunIIFall17DRPremix/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/AODSIM/RECOSIMstep_94X_mc2017_realistic_v10_ext1-v1/80001/8C336D9E-8DFE-E711-981A-0CC47A13D3B2.root',
        '/store/mc/RunIIFall17DRPremix/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/AODSIM/RECOSIMstep_94X_mc2017_realistic_v10_ext1-v1/80001/0040D871-CEFE-E711-8A3B-0CC47A2B04CC.root',
        '/store/mc/RunIIFall17DRPremix/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/AODSIM/RECOSIMstep_94X_mc2017_realistic_v10_ext1-v1/80001/CA38873C-E2FE-E711-BD3B-003048F5B69A.root',
        '/store/mc/RunIIFall17DRPremix/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/AODSIM/RECOSIMstep_94X_mc2017_realistic_v10_ext1-v1/80000/5693479A-01FE-E711-972A-BC305B3909FE.root',
        '/store/mc/RunIIFall17DRPremix/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/AODSIM/RECOSIMstep_94X_mc2017_realistic_v10_ext1-v1/80000/74738E96-01FE-E711-AB3C-BC305B390A59.root',

    	#'/store/mc/RunIIFall17DRPremix/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/AODSIM/RECOSIMstep_94X_mc2017_realistic_v10_ext1-v1/00001/64649185-CAF3-E711-955D-02163E014685.root',
        #'/store/mc/RunIIFall17DRPremix/DYJetsToLL_M-5to50_TuneCP5_13TeV-madgraphMLM-pythia8/AODSIM/PU2017_94X_mc2017_realistic_v11-v3/280000/0002944B-672B-E911-8E3C-0025905B85CC.root',
    ),
)
process.TFileService = cms.Service ('TFileService',
    fileName = cms.string ('images.root')
)

###########################################################
##### Set up the producer and the end path            #####
###########################################################

process.trackImageProducer = cms.EDAnalyzer ("TrackImageProducer",
    tracks       = cms.InputTag("generalTracks"),
    genParticles = cms.InputTag("genParticles", ""),
    electrons    = cms.InputTag("gedGsfElectrons", ""),
    muons        = cms.InputTag("muons", ""),
    taus         = cms.InputTag("hpsPFTauProducer", ""),
    pfCandidates = cms.InputTag("particleFlow", ""),
    vertices     = cms.InputTag("offlinePrimaryVertices", ""),
    jets         = cms.InputTag("ak4PFJetsCHS", ""),

    EBRecHits     =  cms.InputTag("reducedEcalRecHitsEB"),
    EERecHits     =  cms.InputTag("reducedEcalRecHitsEE"),
    ESRecHits     =  cms.InputTag("reducedEcalRecHitsES"),
    HBHERecHits   =  cms.InputTag("reducedHcalRecHits", "hbhereco"),
    CSCSegments   =  cms.InputTag("cscSegments"),
    DTRecSegments =  cms.InputTag("dt4DSegments"),
    RPCRecHits    =  cms.InputTag("rpcRecHits"),

    tauDecayModeFinding      = cms.InputTag("hpsPFTauDiscriminationByDecayModeFinding"),
    tauElectronDiscriminator = cms.InputTag("hpsPFTauDiscriminationByMVA6LooseElectronRejection"),
    tauMuonDiscriminator     = cms.InputTag("hpsPFTauDiscriminationByLooseMuonRejection3"),

    minGenParticlePt = cms.double (10.0),
    minTrackPt       = cms.double (30.0),
    maxRelTrackIso   = cms.double (0.05),
)

process.myPath = cms.Path (process.trackImageProducer)