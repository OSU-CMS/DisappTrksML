"""
Prints a LaTex table to compare number of events passing
fiducial map and deepSets, and ratio of efficiencies.
"""

import ROOT as r
import numpy as np 
import math

# round to first sig fig
def round1(x):
	if x == 0: return 0
	return round(x, -int(math.floor(math.log10(abs(x)))))

def findSigFigIndex(x):
	if x == 0: return 0
	return -int(math.floor(math.log10(abs(x))))

# load the fiducial map sigmas for each data sample
ele_fidMap = np.load("ele_fidMap.npy", allow_pickle=True, encoding="bytes")
h10_fidMap = np.load("h10_fidMap.npy", allow_pickle=True, encoding="bytes")
h100_fidMap = np.load("h100_fidMap.npy", allow_pickle=True, encoding="bytes")
h1000_fidMap = np.load("h1000_fidMap.npy", allow_pickle=True, encoding="bytes")
h10000_fidMap = np.load("h10000_fidMap.npy", allow_pickle=True, encoding="bytes")

# load the deepSets predictions for each data sample
ele_preds = np.load("SingleEle_fullSel_preds.npy", allow_pickle=True, encoding="bytes")
h10_preds = np.load("h10_fullSel_preds.npy", allow_pickle=True, encoding="bytes")
h100_preds = np.load("h100_fullSel_preds.npy", allow_pickle=True, encoding="bytes")
h1000_preds = np.load("h1000_fullSel_preds.npy", allow_pickle=True, encoding="bytes")
h10000_preds = np.load("h10000_fullSel_preds.npy", allow_pickle=True, encoding="bytes")

labels = [r"Tag \& Probe Electrons ",
		"Higgsino 700 GeV, 10cm",
		"Higgsino 700 GeV, 100cm",
		"Higgsino 700 GeV, 1000cm",
		"Higgsino 700 GeV, 10000cm"]
totals = []
npass_clf, totals, npass_fidMap = [], [], []

# cut on deepSets discriminant value disc_values
disc_vaue = 0.5
for file in [ele_preds, h10_preds, h100_preds, h1000_preds, h10000_preds]:
	totals.append(len(file))
	n = len(file[file<disc_vaue])
	npass_clf.append(n)

# cut on fiducial map sigma < 2
for file in [ele_fidMap, h10_fidMap, h100_fidMap, h1000_fidMap, h10000_fidMap]:
	n = len(file[file<2])
	npass_fidMap.append(n)

ratios, ratio_sigmas = [], []
for n1, n2 in zip(npass_clf, npass_fidMap):
    ratios.append(n1*1.0/n2)
    ratio_sigmas.append(np.sqrt(n1*1.0/(n2**2) + (n1**2)*1.0/(n2**3)))

for i in range(len(totals)):
	row = labels[i]
	row += " & "
	row += str(totals[i])
	row += " & "
	n = npass_clf[i]
	s = np.sqrt(n)
	row += str(int(n))
	row += " & "
	n = npass_fidMap[i]
	s = np.sqrt(n)
	row += str(int(n))
	row += " & "
	n = ratios[i]
	s = ratio_sigmas[i]
	if n == 0:
		row += "0^{{+1.1}}_{{-0}}"
	else:
		row += r"{0}\pm{1}".format(round(n, findSigFigIndex(s)), round1(s))
	row += r" \\"
	print(row)
