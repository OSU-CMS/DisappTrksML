import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import utils
#import ROOT as r
#from ROOT import TH1F, TCanvas


if len(sys.argv) > 1: filenameE = str(sys.argv[1])
else: filenameE = "falseEventsE.npy"
if len(sys.argv) > 2: filenameB = str(sys.argv[2])
else: filenameB = "falseEventsB.npy"

imageDir = "/store/user/mcarrigan/disappearingTracks/electron_selection_ZtoEE_All/"
dataDir = "/data/users/mcarrigan/cnn_9_17_ZtoEE_All_V2/cnn_9_17_ZtoEE_All_V2_p0/outputFiles/"
nameStartE = "e_0p25_"
nameStartB = "bkg_0p25_"
tag = ""
saveDir = '/home/mcarrigan/scratch0/disTracksML/plots/falseEvents/ZtoEE_All/'

if not os.path.exists(saveDir): os.mkdir(saveDir) 

filenameE = dataDir + filenameE
filenameB = dataDir + filenameB

events_e = np.load(filenameE, allow_pickle=True)
events_e = np.reshape(events_e, (-1,4))
these_events = np.array(np.where(events_e[:,0] == 2621)).flatten()
print(events_e[these_events])
events_e = events_e[events_e[:,0].argsort()]
#getting only events >95% electron or < 0.05%
positive_ele = np.array(np.where(events_e[:,3] > 0.95)).flatten()
negative_ele = np.array(np.where(events_e[:,3] < 0.5)).flatten()
posE = events_e[positive_ele]
negE = events_e[negative_ele]
pos_fileE = posE[:, 0]
pos_indexE = posE[:, 1]
pos_predE = posE[:, 3]
pos_predE = np.concatenate(pos_predE).flatten()
neg_fileE = negE[:, 0]
neg_indexE = negE[:, 1]
neg_predE = negE[:, 3]
neg_predE = np.concatenate(neg_predE).flatten()

false_indices = np.where(events_e[:,3] < 0.5)
events_e = events_e[false_indices]
file_e = events_e[:, 0]
index_e = events_e[:, 1]
truth_e = events_e[:, 2]
pred_e = events_e[:, 3]
pred_e = np.concatenate(pred_e).flatten()

events_b = np.load(filenameB, allow_pickle=True)
events_b = np.reshape(events_b, (-1, 4))
events_b = events_b[events_b[:,0].argsort()]
#getting only events >95% background or < 0.05%
positive_bkg = np.array(np.where(events_b[:,3] > 0.95)).flatten()
negative_bkg = np.array(np.where(events_b[:,3] < 0.1)).flatten()
posB = events_b[positive_bkg]
negB = events_b[negative_bkg]
pos_fileB = posB[:, 0]
pos_indexB = posB[:, 1]
pos_predB = posB[:, 3]
pos_predB = np.concatenate(pos_predB).flatten()
neg_fileB = negB[:, 0]
neg_indexB = negB[:, 1]
neg_predB = negB[:, 3]
neg_predB = np.concatenate(neg_predB).flatten()

false_indices = np.where(events_b[:,3] >= 0.5)
events_b = events_b[false_indices]
file_b = events_b[:, 0]
index_b = events_b[:, 1]
truth_b = events_b[:, 2]
pred_b = events_b[:, 3]
if pred_b.shape[0] !=0: pred_b = np.concatenate(pred_b).flatten()

fig1, hPred = plt.subplots(1,2, figsize = (20,5))
print("Making Histograms")
hPred[0].hist(pred_e, bins = 25)
hPred[0].set_title("Prediction Score of Electron Events Incorrectly Reconstructed")
hPred[0].set(xlabel = "Prediction Score")
hPred[1].hist(pred_b, bins = 25)
hPred[1].set_title("Prediction Score of Background Events Incorrectly Reconstructed")
hPred[1].set(xlabel = "Prediction Score")
plt.savefig(saveDir + "PredScores.png")
plt.close()

#c1 = TCanvas("c1", "canvas1", 600, 800)
#h_ID = TH1F("h_ID", "PDG ID of Misidentified Particles", 500, 0, 500)

def plotFile(files, index, pred, nameStart, dataDir, imageDir, saveDir, savetag):
    start_file = -1
    file_events = []
    evt_counter = 0
    eta = []
    phi = []
    hcal =[]
    ecal = []
    muons = []
    preshower = []
    files = np.append(files, -1)
    for event in range(len(files)):
        if event == 0: start_file = int(files[event])
        #if start_file > 500: continue
        if start_file == files[event]: file_events.append(index[event])
        if start_file != files[event] or event == len(files)-1:
            filename = nameStart + str(start_file) + tag
            data, info = utils.load_electron_data(imageDir, filename)
            data = data[:,2:]
            data = np.reshape(data, [len(data),40,40,4])
            ID = info[:,1]
            row = int(0)
            if len(file_events) <= 25 :
                plot_len = len(file_events) / 5 + 1
                plot_len = int(plot_len)
                if plot_len < 2: plot_len = 2
                fig2, e_img = plt.subplots(plot_len, 5, figsize = (25, plot_len*5))
                for j in range(len(file_events)):
                    this_index = np.where(ID==file_events[j])[0][0]
                    eta.append(info[this_index,7])
                    phi.append(info[this_index,8])
                    hcal.append(np.sum(data[this_index, :, :, 2]))
                    preshower.append(np.sum(data[this_index, :, :, 1]))
                    muons.append(np.sum(data[this_index, :, :, 3]))
                    ecal.append(np.sum(data[this_index, :, :, 0]))
                    if j != 0 and j%5 == 0: row += 1
                    col = int(j%5)
                    #h_ID.Fill(info[this_index,15])
                    e_img[row, col].imshow(np.arctanh(data[this_index, :, :, 0]))
                    e_img[row, col].set_title("event: " + str(file_events[j]) + " prob: " + str(pred[evt_counter]))
                    e_img[row, col].annotate("eta = "+str(info[this_index,7]), xy=(0,0), xytext=(25,3), color='white')
                    e_img[row, col].annotate("phi = "+str(info[this_index,8]), xy=(0,0), xytext=(25,5), color='white')
                    evt_counter += 1
                this_type = nameStart.split("_")
                this_type = this_type[0]
                plt.savefig(saveDir + savetag + this_type + str(start_file) + ".png")
                plt.close()
                if event == len(files)-1:
                    continue
                start_file = int(files[event])
                file_events = [index[event]]

            else:
                this_type = nameStart.split("_")
                this_type = this_type[0]
                counter = 0
                fig2, e_img = plt.subplots(5, 5, figsize = (25, 25))
                for j in range(len(file_events)):
                    this_index = np.where(ID==file_events[j])[0][0]
                    eta.append(info[this_index,7])
                    phi.append(info[this_index,8])
                    hcal.append(np.sum(data[this_index, :, :, 2]))
                    preshower.append(np.sum(data[this_index, :, :, 1]))
                    ecal.append(np.sum(data[this_index, :, :, 0]))
                    muons.append(np.sum(data[this_index, :, :, 3]))
                    if j != 0 and j%5 == 0: row += 1
                    if j != 0 and j%25 == 0:
                       plt.savefig(saveDir + savetag + this_type + str(start_file) +"_"+str(counter)+ ".png")
                       counter += 1
                       row = 0
                    col = int(j%5)
                    e_img[row, col].imshow(np.arctanh(data[this_index, :, :, 0]))
                    e_img[row, col].set_title("pred: " + str(pred[evt_counter]))
                    e_img[row, col].annotate("eta = "+str(info[this_index,7]), xy=(0,0), xytext=(25,3), color='white')
                    e_img[row, col].annotate("phi = "+str(info[this_index,8]), xy=(0,0), xytext=(25,5), color='white')
                    evt_counter += 1
                plt.savefig(saveDir + savetag + this_type + str(start_file) +"_"+str(counter)+ ".png")
                plt.close()
                if event == len(files)-1:
                    continue
                start_file = int(files[event])
                file_events = [index[event]]
    fig3, h_etaPhi = plt.subplots(1,2, figsize=(10,5))
    h_etaPhi[0].hist(eta, bins = 40)
    h_etaPhi[0].set_title("Eta of Misidentified Events")
    h_etaPhi[0].set(xlabel='Eta')
    h_etaPhi[1].hist(phi, bins = 40)             
    h_etaPhi[1].set_title("Phi of Misidentified Events")
    h_etaPhi[1].set(xlabel='Phi')
    plt.savefig(saveDir + nameStart+ "failedImages_EtaPhi.png")
    plt.close()
    fig4, h_numHits = plt.subplots(1, 4, figsize=(15, 4))
    h_numHits[0].hist(ecal, bins=20)
    h_numHits[0].set_title("Energy of ECAL Event")
    h_numHits[0].set(xlabel="Energy")
    h_numHits[1].hist2d(ecal, preshower, bins=[20,20])
    h_numHits[1].set_title("Energy of ECAL vs Preshower in Events")
    h_numHits[1].set(xlabel="ECAL Energy")
    h_numHits[1].set(ylabel="Preshower Energy")
    h_numHits[2].hist2d(ecal, hcal, bins=[20,20])
    h_numHits[2].set_title("Energy of ECAL vs HCAL in Events")
    h_numHits[2].set(xlabel="ECAL Energy")
    h_numHits[2].set(ylabel="HCAL Energy")
    h_numHits[3].hist2d(ecal, muons, bins=[20,20])
    h_numHits[3].set_title("Energy of ECAL vs Muon in Events")
    h_numHits[3].set(xlabel="ECAL Energy")
    h_numHits[3].set(ylabel="Muons Energy")
    plt.savefig(saveDir+savetag+"numHits.png")
print("Saving False Events")
#plotFile(pos_fileE, pos_indexE, pos_predE, nameStartE, dataDir, imageDir, saveDir, "posReco_")
plotFile(neg_fileE, neg_indexE, neg_predE, nameStartE, dataDir, imageDir, saveDir, "negReco_")
#plotFile(pos_fileB, pos_indexB, pos_predB, nameStartB, dataDir, imageDir, saveDir, "posReco_")
#plotFile(neg_fileB, neg_indexB, neg_predB, nameStartB, dataDir, imageDir, saveDir, "negReco_")
#plotFile(file_e, index_e, pred_e, nameStartE, dataDir, imageDir, saveDir, "negReco_")


