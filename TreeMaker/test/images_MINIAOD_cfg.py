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
process.GlobalTag = GlobalTag(process.GlobalTag, '94X_dataRun2_ReReco_EOY17_v6', '')

process.maxEvents = cms.untracked.PSet (
    input = cms.untracked.int32 (100)
)
process.source = cms.Source ("PoolSource",
    fileNames = cms.untracked.vstring (
        'root://xrootd.rcac.purdue.edu//store/user/bfrancis/SingleElectron/Run2017F-31Mar2018-v1-DisappTrks-v3/200803_182351/0000/REMINIAOD_PAT_547.root',
    ),
)
process.TFileService = cms.Service ('TFileService',
    fileName = cms.string ('images.root')
)

###########################################################
##### Set up the producer and the end path            #####
###########################################################

process.trackImageProducer = cms.EDAnalyzer ("TrackImageProducerMINIAOD",
    triggers       = cms.InputTag("TriggerResults", "", "HLT"),
    triggerObjects = cms.InputTag("slimmedPatTrigger"),
    tracks         = cms.InputTag("candidateTrackProducer"),
    genParticles   = cms.InputTag("prunedGenParticles", ""),
    met            = cms.InputTag("slimmedMETs"),
    electrons      = cms.InputTag("slimmedElectrons", ""),
    muons          = cms.InputTag("slimmedMuons", ""),
    taus           = cms.InputTag("slimmedTaus", ""),
    pfCandidates   = cms.InputTag("packedPFCandidates", ""),
    vertices       = cms.InputTag("offlineSlimmedPrimaryVertices", ""),
    jets           = cms.InputTag("slimmedJets", ""),

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

    minGenParticlePt = cms.double(10.0),
    minTrackPt       = cms.double(30.0),
    maxRelTrackIso   = cms.double(0.05),

    dataTakingPeriod = cms.string("2017")
)

process.myPath = cms.Path (process.trackImageProducer)