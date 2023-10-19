# DisappTrksML

## Recipie for 2022 data

```
cmsrel CMSSW_12_4_11_patch3
cd CMSSW_12_4_11_patch3/src
cmsenv
git-cms-init
git clone https://github.com/OSU-CMS/DisappTrksML.git
scramv1 b -j 9
cd DisappTrksML/TreeMaker/test
cmsRun treeMaker_MC2022_cfg.py
```

## Recipe for 2017 data

```
cmsrel CMSSW_9_4_9
cd CMSSW_9_4_9/src
cmsenv
git clone https://github.com/OSU-CMS/DisappTrksML.git
scramv1 b -j 9
cd DisappTrksML/TreeMaker/test
cmsRun images_cfg.py
```

## 'Full' recipe including osusub.py

```
cmsrel CMSSW_9_4_9
cd CMSSW_9_4_9/src/
cmsenv
git cms-merge-topic cms-met:METFixEE2017_949_v2
git cms-merge-topic UAEDF-tomc:eleCutBasedId_94X_V2
git clone https://github.com/OSU-CMS/OSUT3Analysis.git
git clone https://github.com/OSU-CMS/DisappTrks.git
git clone https://github.com/OSU-CMS/DisappTrksML.git
OSUT3Analysis/AnaTools/scripts/setupFramework.py -f MINI_AOD_2017 -c DisappTrks/StandardAnalysis/interface/CustomDataFormat.h
scram b -j 9
cmsenv
```
