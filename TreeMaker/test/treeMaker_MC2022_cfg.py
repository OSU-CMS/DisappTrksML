# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: step1 --filein dbs:/ZJetsToNuNu_HT-100To200_13TeV-madgraph/RunIIFall17MiniAODv2-PU2017_12Apr2018_94X_mc2017_realistic_v14-v1/MINIAODSIM --mc --eventcontent MINIAODSIM --runUnscheduled --datatier MINIAODSIM --conditions 94X_mc2017_realistic_v15 --step PAT --nThreads 4 --scenario pp --era Run2_2017,run2_miniAOD_94XFall17 --python_filename candidateTrackProducer_RunMiniAOD_MC2017_cfg.py --no_exec --customise Configuration/DataProcessing/Utils.addMonitoring -n 100
import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras
#from PhysicsTools.PatAlgos.slimming.isolatedTracks_cfi import *

process = cms.Process('TreeMaker',eras.Run3)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('PhysicsTools.PatAlgos.slimming.metFilterPaths_cff')
process.load('Configuration.StandardSequences.PATMC_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
      'root://cmsxrootd.fnal.gov:///store/mc/Run3Summer22EEMiniAODv3/DYToLL_M-50_TuneCP5_13p6TeV-pythia8/MINIAODSIM/Poisson60KeepRAW_124X_mcRun3_2022_realistic_postEE_v1-v2/2560000/0143a7e0-a7ca-4667-b3e2-33b322f2e6cc.root'
    ),
    secondaryFileNames = cms.untracked.vstring(
      'root://cmsxrootd.fnal.gov://store/mc/Run3Summer22EEDR/DYToLL_M-50_TuneCP5_13p6TeV-pythia8/AODSIM/Poisson60KeepRAW_124X_mcRun3_2022_realistic_postEE_v1-v2/2560000/01eee9fb-a66a-452a-95ec-1062ed94640b.root',
      'root://cmsxrootd.fnal.gov://store/mc/Run3Summer22EEDR/DYToLL_M-50_TuneCP5_13p6TeV-pythia8/AODSIM/Poisson60KeepRAW_124X_mcRun3_2022_realistic_postEE_v1-v2/2560000/185cc6fd-68cc-4b8d-89ad-e67ca8b10914.root',
      'root://cmsxrootd.fnal.gov://store/mc/Run3Summer22EEDR/DYToLL_M-50_TuneCP5_13p6TeV-pythia8/AODSIM/Poisson60KeepRAW_124X_mcRun3_2022_realistic_postEE_v1-v2/2560000/52155bde-4fbd-42d7-a7eb-399c3836d553.root',
      'root://cmsxrootd.fnal.gov://store/mc/Run3Summer22EEDR/DYToLL_M-50_TuneCP5_13p6TeV-pythia8/AODSIM/Poisson60KeepRAW_124X_mcRun3_2022_realistic_postEE_v1-v2/2560000/61c2356c-2f7c-4c20-a777-25c59990e61b.root',
      'root://cmsxrootd.fnal.gov://store/mc/Run3Summer22EEDR/DYToLL_M-50_TuneCP5_13p6TeV-pythia8/AODSIM/Poisson60KeepRAW_124X_mcRun3_2022_realistic_postEE_v1-v2/2560000/85fdfb16-ce76-4b4b-b81b-c81d91d1eca6.root',
      'root://cmsxrootd.fnal.gov://store/mc/Run3Summer22EEDR/DYToLL_M-50_TuneCP5_13p6TeV-pythia8/AODSIM/Poisson60KeepRAW_124X_mcRun3_2022_realistic_postEE_v1-v2/2560000/b5c830cd-82c4-462b-9dd1-a23ceb4c83ce.root',
      'root://cmsxrootd.fnal.gov://store/mc/Run3Summer22EEDR/DYToLL_M-50_TuneCP5_13p6TeV-pythia8/AODSIM/Poisson60KeepRAW_124X_mcRun3_2022_realistic_postEE_v1-v2/2560000/c035571b-650e-4f09-aaaa-aae6ac1ba30a.root',
      'root://cmsxrootd.fnal.gov://store/mc/Run3Summer22EEDR/DYToLL_M-50_TuneCP5_13p6TeV-pythia8/AODSIM/Poisson60KeepRAW_124X_mcRun3_2022_realistic_postEE_v1-v2/2560000/db93f101-423e-4ad7-bfe3-67731c4979a8.root',
      'root://cmsxrootd.fnal.gov://store/mc/Run3Summer22EEDR/DYToLL_M-50_TuneCP5_13p6TeV-pythia8/AODSIM/Poisson60KeepRAW_124X_mcRun3_2022_realistic_postEE_v1-v2/2560000/ea0bf351-9ce1-4548-a385-dce72855d42e.root',
      'root://cmsxrootd.fnal.gov://store/mc/Run3Summer22EEDR/DYToLL_M-50_TuneCP5_13p6TeV-pythia8/AODSIM/Poisson60KeepRAW_124X_mcRun3_2022_realistic_postEE_v1-v2/2560000/6a95414b-7c14-4262-875a-7749ee59561b.root',    ),
)


process.TFileService = cms.Service ('TFileService',
    fileName = cms.string ('images.root')
)

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('step1 nevts:4800'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

process.MINIAODSIMoutput = cms.OutputModule("PoolOutputModule",
    compressionAlgorithm = cms.untracked.string('LZMA'),
    compressionLevel = cms.untracked.int32(4),
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('MINIAODSIM'),
        filterName = cms.untracked.string('')
    ),
    dropMetaData = cms.untracked.string('ALL'),
    eventAutoFlushCompressedSize = cms.untracked.int32(-900),
    fastCloning = cms.untracked.bool(False),
    fileName = cms.untracked.string('step1_PAT.root'),
    outputCommands = process.MINIAODSIMEventContent.outputCommands,
    overrideBranchesSplitLevel = cms.untracked.VPSet(cms.untracked.PSet(
        branch = cms.untracked.string('patPackedCandidates_packedPFCandidates__*'),
        splitLevel = cms.untracked.int32(99)
    ), 
        cms.untracked.PSet(
            branch = cms.untracked.string('recoGenParticles_prunedGenParticles__*'),
            splitLevel = cms.untracked.int32(99)
        ), 
        cms.untracked.PSet(
            branch = cms.untracked.string('patTriggerObjectStandAlones_slimmedPatTrigger__*'),
            splitLevel = cms.untracked.int32(99)
        ), 
        cms.untracked.PSet(
            branch = cms.untracked.string('patPackedGenParticles_packedGenParticles__*'),
            splitLevel = cms.untracked.int32(99)
        ), 
        cms.untracked.PSet(
            branch = cms.untracked.string('patJets_slimmedJets__*'),
            splitLevel = cms.untracked.int32(99)
        ), 
        cms.untracked.PSet(
            branch = cms.untracked.string('recoVertexs_offlineSlimmedPrimaryVertices__*'),
            splitLevel = cms.untracked.int32(99)
        ), 
        cms.untracked.PSet(
            branch = cms.untracked.string('recoCaloClusters_reducedEgamma_reducedESClusters_*'),
            splitLevel = cms.untracked.int32(99)
        ), 
        cms.untracked.PSet(
            branch = cms.untracked.string('EcalRecHitsSorted_reducedEgamma_reducedEBRecHits_*'),
            splitLevel = cms.untracked.int32(99)
        ), 
        cms.untracked.PSet(
            branch = cms.untracked.string('EcalRecHitsSorted_reducedEgamma_reducedEERecHits_*'),
            splitLevel = cms.untracked.int32(99)
        ), 
        cms.untracked.PSet(
            branch = cms.untracked.string('recoGenJets_slimmedGenJets__*'),
            splitLevel = cms.untracked.int32(99)
        ), 
        cms.untracked.PSet(
            branch = cms.untracked.string('patJets_slimmedJetsPuppi__*'),
            splitLevel = cms.untracked.int32(99)
        ), 
        cms.untracked.PSet(
            branch = cms.untracked.string('EcalRecHitsSorted_reducedEgamma_reducedESRecHits_*'),
            splitLevel = cms.untracked.int32(99)
        )),
    overrideInputFileSplitLevels = cms.untracked.bool(True),
    splitLevel = cms.untracked.int32(0)
)

# Additional output definition

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '124X_dataRun3_Prompt_v4', '')

#process.load('DisappTrks.CandidateTrackProducer.CandidateTrackProducer_cfi')
#process.candidateTracks = cms.Path(process.candidateTrackProducer)
#from DisappTrks.CandidateTrackProducer.customize import disappTrksOutputCommands
#process.MINIAODSIMoutput.outputCommands.extend(disappTrksOutputCommands)

process.trackImageProducer = cms.EDAnalyzer ("TrackImageProducerMINIAOD",
    triggers	   = cms.InputTag("TriggerResults", "", "HLT"),
    triggerObjects = cms.InputTag("slimmedPatTrigger"),
    tracks         = cms.InputTag("candidateTrackProducer"),
    genParticles   = cms.InputTag("prunedGenParticles", ""),
    met            = cms.InputTag("slimmedMETs"),
    electrons	   = cms.InputTag("slimmedElectrons", ""),
    muons          = cms.InputTag("slimmedMuons", ""),
    taus           = cms.InputTag("slimmedTaus", ""),
    pfCandidates   = cms.InputTag("packedPFCandidates", ""),
    vertices	   = cms.InputTag("offlineSlimmedPrimaryVertices", ""),
    jets           = cms.InputTag("slimmedJets", ""),

    rhoCentralCalo = cms.InputTag("fixedGridRhoFastjetCentralCalo"),

    EBRecHits     =  cms.InputTag("reducedEcalRecHitsEB"),
    EERecHits     =  cms.InputTag("reducedEcalRecHitsEE"),
    ESRecHits     =  cms.InputTag("reducedEcalRecHitsES"),
    HBHERecHits   =  cms.InputTag("reducedHcalRecHits", "hbhereco"),
    #CSCSegments   =  cms.InputTag("cscSegments"),
    #DTRecSegments =  cms.InputTag("dt4DSegments"),
    #RPCRecHits    =  cms.InputTag("rpcRecHits"),

    tauDecayModeFinding      = cms.InputTag("hpsPFTauDiscriminationByDecayModeFinding"),
    tauElectronDiscriminator = cms.InputTag("hpsPFTauDiscriminationByMVA6LooseElectronRejection"),
    tauMuonDiscriminator     = cms.InputTag("hpsPFTauDiscriminationByLooseMuonRejection3"),
    
    dEdxPixel = cms.InputTag ("dedxPixelHarmonic2", ""),
    dEdxStrip = cms.InputTag ("dedxHarmonic2", ""),
    isolatedTracks = cms.InputTag("isolatedTracks", ""),
    isoTrk2dedxHitInfo = cms.InputTag("isolatedTracks", ""),
    genTracks = cms.InputTag("generalTracks", ""),
    pileupInfo = cms.InputTag ("addPileupInfo"),

    signalTriggerNames = cms.vstring([
	'HLT_MET105_IsoTrk50_v',
	'HLT_PFMET120_PFMHT120_IDTight_v',
        'HLT_PFMET130_PFMHT130_IDTight_v',
        'HLT_PFMET140_PFMHT140_IDTight_v',
        'HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v',
        'HLT_PFMETNoMu130_PFMHTNoMu130_IDTight_v',
        'HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_v',
        'HLT_PFMET250_HBHECleaned_v',
        'HLT_PFMET300_HBHECleaned_v']),

    metFilterNames = cms.vstring([
        "Flag_goodVertices",
        "Flag_globalTightHalo2016Filter",
        "Flag_HBHENoiseFilter",
        "Flag_HBHENoiseIsoFilter",
        "Flag_EcalDeadCellTriggerPrimitiveFilter",
	"Flag_BadPFMuonFilter",
	"Flag_globalTightHalo2016Filter",
	"Flag_globalSuperTightHalo2016Filter"]),

    minGenParticlePt = cms.double(10),
    minTrackPt       = cms.double(25.0),
    maxRelTrackIso   = cms.double(-1.0),
    maxTrackEta = cms.double(2.1),

    dataTakingPeriod = cms.string("2022")
)
process.trackImageProducerPath = cms.Path(process.trackImageProducer)

process.MINIAODSIMoutput.outputCommands = cms.untracked.vstring('drop *')

process.isolatedTracks.saveDeDxHitInfoCut = cms.string("pt > 1.")

# Path and EndPath definitions
process.Flag_trackingFailureFilter = cms.Path(process.goodVertices+process.trackingFailureFilter)
process.Flag_goodVertices = cms.Path(process.primaryVertexFilter)
process.Flag_CSCTightHaloFilter = cms.Path(process.CSCTightHaloFilter)
process.Flag_trkPOGFilters = cms.Path(process.trkPOGFilters)
process.Flag_HcalStripHaloFilter = cms.Path(process.HcalStripHaloFilter)
process.Flag_trkPOG_logErrorTooManyClusters = cms.Path(~process.logErrorTooManyClusters)
process.Flag_EcalDeadCellTriggerPrimitiveFilter = cms.Path(process.EcalDeadCellTriggerPrimitiveFilter)
process.Flag_ecalLaserCorrFilter = cms.Path(process.ecalLaserCorrFilter)
process.Flag_globalSuperTightHalo2016Filter = cms.Path(process.globalSuperTightHalo2016Filter)
process.Flag_eeBadScFilter = cms.Path(process.eeBadScFilter)
process.Flag_METFilters = cms.Path(process.metFilters)
process.Flag_chargedHadronTrackResolutionFilter = cms.Path(process.chargedHadronTrackResolutionFilter)
process.Flag_globalTightHalo2016Filter = cms.Path(process.globalTightHalo2016Filter)
process.Flag_CSCTightHaloTrkMuUnvetoFilter = cms.Path(process.CSCTightHaloTrkMuUnvetoFilter)
process.Flag_HBHENoiseIsoFilter = cms.Path(process.HBHENoiseFilterResultProducer+process.HBHENoiseIsoFilter)
process.Flag_BadChargedCandidateSummer16Filter = cms.Path(process.BadChargedCandidateSummer16Filter)
process.Flag_hcalLaserEventFilter = cms.Path(process.hcalLaserEventFilter)
process.Flag_BadPFMuonFilter = cms.Path(process.BadPFMuonFilter)
process.Flag_ecalBadCalibFilter = cms.Path(process.ecalBadCalibFilter)
process.Flag_HBHENoiseFilter = cms.Path(process.HBHENoiseFilterResultProducer+process.HBHENoiseFilter)
process.Flag_trkPOG_toomanystripclus53X = cms.Path(~process.toomanystripclus53X)
process.Flag_EcalDeadCellBoundaryEnergyFilter = cms.Path(process.EcalDeadCellBoundaryEnergyFilter)
process.Flag_BadChargedCandidateFilter = cms.Path(process.BadChargedCandidateFilter)
process.Flag_trkPOG_manystripclus53X = cms.Path(~process.manystripclus53X)
process.Flag_BadPFMuonSummer16Filter = cms.Path(process.BadPFMuonSummer16Filter)
process.Flag_muonBadTrackFilter = cms.Path(process.muonBadTrackFilter)
process.Flag_CSCTightHalo2015Filter = cms.Path(process.CSCTightHalo2015Filter)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.MINIAODSIMoutput_step = cms.EndPath(process.MINIAODSIMoutput)

# Schedule definition
'''process.schedule = cms.Schedule(
    process.Flag_HBHENoiseFilter,
    process.Flag_HBHENoiseIsoFilter,
    process.Flag_CSCTightHaloFilter,
    process.Flag_CSCTightHaloTrkMuUnvetoFilter,
    process.Flag_CSCTightHalo2015Filter,
    process.Flag_globalTightHalo2016Filter,
    process.Flag_globalSuperTightHalo2016Filter,
    process.Flag_HcalStripHaloFilter,
    process.Flag_hcalLaserEventFilter,
    process.Flag_EcalDeadCellTriggerPrimitiveFilter,
    process.Flag_EcalDeadCellBoundaryEnergyFilter,
    process.Flag_ecalBadCalibFilter,
    process.Flag_goodVertices,
    process.Flag_eeBadScFilter,
    process.Flag_ecalLaserCorrFilter,
    process.Flag_trkPOGFilters,
    process.Flag_chargedHadronTrackResolutionFilter,
    process.Flag_muonBadTrackFilter,
    process.Flag_BadChargedCandidateFilter,
    process.Flag_BadPFMuonFilter,
    process.Flag_BadChargedCandidateSummer16Filter,
    process.Flag_BadPFMuonSummer16Filter,
    process.Flag_trkPOG_manystripclus53X,
    process.Flag_trkPOG_toomanystripclus53X,
    process.Flag_trkPOG_logErrorTooManyClusters,
    process.Flag_METFilters,
    process.candidateTracks,
    process.trackImageProducerPath,
    process.endjob_step,
    process.MINIAODSIMoutput_step)
process.schedule.associate(process.patTask)
from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
associatePatAlgosToolsTask(process)
'''

process.schedule = cms.Schedule(process.trackImageProducerPath)

#Setup FWK for multithreaded
process.options.numberOfThreads=cms.untracked.uint32(4)
process.options.numberOfStreams=cms.untracked.uint32(0)
process.options.numberOfConcurrentLuminosityBlocks=cms.untracked.uint32(1)
# customisation of the process.
process.options.SkipEvent=cms.untracked.vstring("trackImageProducerPath")

# Automatic addition of the customisation function from Configuration.DataProcessing.Utils
from Configuration.DataProcessing.Utils import addMonitoring 

#call to customisation function addMonitoring imported from Configuration.DataProcessing.Utils
process = addMonitoring(process)

# End of customisation functions
#do not add changes to your config after this point (unless you know what you are doing)
from FWCore.ParameterSet.Utilities import convertToUnscheduled
process=convertToUnscheduled(process)

# customisation of the process.

# Automatic addition of the customisation function from PhysicsTools.PatAlgos.slimming.miniAOD_tools
from PhysicsTools.PatAlgos.slimming.miniAOD_tools import miniAOD_customizeAllMC 

#call to customisation function miniAOD_customizeAllMC imported from PhysicsTools.PatAlgos.slimming.miniAOD_tools
process = miniAOD_customizeAllMC(process)

# End of customisation functions

# Customisation from command line

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion
