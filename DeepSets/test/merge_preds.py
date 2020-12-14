import os, sys
import numpy as np 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

dataDir = '/share/scratch0/llavezzo/CMSSW_11_1_3/src/DisappTrksML/DeepSets/test/preds_SingleMu2017/'

preds = []

for file in os.listdir(dataDir):
	if ".root.npz" not in file and "preds" not in file: continue
	indata = np.load(dataDir+file,allow_pickle=True)
	p = indata['preds']
	if(len(p)==0):continue
	for pr in p: preds.append(pr)


preds = np.array(preds)
count = np.count_nonzero(preds>0.5)

print("Found",count,"muons from",len(preds),"true muons")

plt.hist(preds)
plt.xlabel("Classifier Output")
plt.savefig("classifier_ouput_SingleMu2017F.png")