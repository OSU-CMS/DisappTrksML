#!/usr/bin/env python

import os
import sys
import CRABClient
from CRABClient.UserUtilities import config
config = config()

config.General.requestName = ''
config.General.workArea = 'crab'
config.General.transferOutputs = True
config.General.transferLogs = True

config.JobType.pluginName = 'Analysis'
config.JobType.psetName = 'treeMaker_MC2022_cfg.py'  # For MC only
config.JobType.allowUndistributedCMSSW = True

config.JobType.numCores = 4
config.JobType.maxMemoryMB = 2500
config.Data.inputDataset = '/DYJetsToLL_M-50_TuneCP5_13p6TeV-madgraphMLM-pythia8/Run3Summer22MiniAODv3-forPOG_124X_mcRun3_2022_realistic_v12-v4/MINIAODSIM'
config.Data.secondaryInputDataset = '/DYJetsToLL_M-50_TuneCP5_13p6TeV-madgraphMLM-pythia8/Run3Summer22DRPremix-forPOG_124X_mcRun3_2022_realistic_v12-v4/AODSIM'
config.Data.useParent = False
config.Data.inputDBS = 'global'
config.Data.splitting = 'LumiBased' # for both MC and data
config.Data.outLFNDirBase = '/store/user/micarrig/'
config.Data.publication = True
config.Data.outputDatasetTag = ''
#config.Data.lumiMask = 'crab/crab_20230511_111457/results/notFinishedLumis.json'

config.Site.storageSite = 'T2_US_Purdue'

config.Data.outputDatasetTag = 'Images-v1p2-DYJets-madgraph-MC2022'
config.Data.unitsPerJob = 100 # 61275 lumis
config.General.requestName = 'treeMakerNN_DYJetsToLL_M-50_madgraph_v1p2_MC2022'

if __name__ == '__main__':

    from CRABAPI.RawCommand import crabCommand
    from CRABClient.ClientExceptions import ClientException
    from multiprocessing import Process

    def submit(config):
        crabCommand('submit', config = config)

    def forkAndSubmit(config):
        p = Process(target=submit, args=(config,))
        p.start()
        p.join()


    #############################################################################################
    ## From now on that's what users should modify: this is the a-la-CRAB2 configuration part. ##
    #############################################################################################

    #############################################################################################
    ## Data
    #############################################################################################

    #config.Data.lumiMask = 'https://cms-service-dqm.web.cern.ch/cms-service-dqm/CAF/certification/Collisions17/13TeV/ReReco/Cert_294927-306462_13TeV_EOY2017ReReco_Collisions17_JSON_v1.txt'

    # The AOD re-reco is labeled 17Nov17, however the re-reco campaign is labeled 31Mar2018

    # Run2017F-v1

    config.Data.outputDatasetTag = 'Images-v1p3-DYJets-madgraph-MC2022'
    config.Data.unitsPerJob = 100 # 61275 lumis

    config.General.requestName = 'treeMakerNN_DYJetsToLL_M-50_madgraph_v1p3_MC2022'
    #config.Data.inputDataset   = '/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/RunIIFall17DRPremix-RECOSIMstep_94X_mc2017_realistic_v10-v1/AODSIM'
    #config.JobType.psetName    = 'treeMaker_MC2017_cfg.py'
    forkAndSubmit(config)
