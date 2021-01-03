import os, sys
import numpy as np 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

dataDir = '/store/user/llavezzo/disappearingTracks/SingleEle2017F_validation/trainV2_param5_lessepochs/'

preds = []

for file in os.listdir(dataDir):
	if ".root.npz" not in file and "preds" not in file: continue
	indata = np.load(dataDir+file,allow_pickle=True)
	p = indata['preds']
	if(len(p)==0):continue
	for pr in p: preds.append(pr)

preds = np.array(preds)
count = np.count_nonzero(preds>0.5)

print("Found",count,"electrons from",len(preds),"true electrons")

plt.hist(preds)
plt.xlabel("Classifier Output")
plt.savefig("trainV2_param5_lessepochs/classifier_ouput_SingleEle2017F.png")