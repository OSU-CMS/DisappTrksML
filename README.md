# DisappTrksML

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
