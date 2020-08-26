import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import utils


if len(sys.argv) > 1: filenameE = str(sys.argv[1])
else: print("need to specify electron input file")
if len(sys.argv) > 2: filenameB = str(sys.argv[2])
else:print("need to specify background input file")

imageDir = "/store/user/mcarrigan/disappearingTracks/electron_selection_tanh_5gt0p5/"
dataDir = "/home/mcarrigan/scratch0/disTracksML/DisappTrksML/CNN/cnn_valTest/outputFiles/"
nameStartE = "e_0p25_"
nameStartB = "bkg_0p25_"
tag = ""
saveDir = '/home/mcarrigan/scratch0/disTracksML/plots/falseEvents/valTest_5gt0p5/'

filenameE = dataDir + filenameE
filenameB = dataDir + filenameB

events_e = np.load(filenameE, allow_pickle=True)
events_e = np.reshape(events_e, (-1,4))
file_e = events_e[:, 0]
index_e = events_e[:, 1]
truth_e = events_e[:, 2]
pred_e = events_e[:, 3]

events_b = np.load(filenameB, allow_pickle=True)
events_b = np.reshape(events_b, (-1, 4))
file_b = events_b[:, 0]
index_b = events_b[:, 1]
truth_b = events_b[:, 2]
pred_b = events_b[:, 3]

fig1, hPred = plt.subplots(1,2, figsize = (20,5))

hPred[0].hist(pred_e, bins = 20)
hPred[0].set_title("Prediction Score of Electron Events Incorrectly Reconstructed")
hPred[0].set(xlabel = "Prediction Score")
hPred[1].hist(pred_b, bins = 50)
hPred[1].set_title("Prediction Score of Background Events Incorrectly Reconstructed")
hPred[1].set(xlabel = "Prediction Score")
plt.savefig(saveDir + "PredScores.png")
plt.close()

def plotFile(files, index, pred, nameStart, dataDir, imageDir, saveDir):
    start_file = -1
    file_events = []
    evt_counter = 0
    eta = []
    phi = []
    files = np.append(files, -1)
    for event in range(len(files)):
        if event == 0: start_file = int(files[event])
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
                    if j != 0 and j%5 == 0: row += 1
                    col = int(j%5)
                    e_img[row, col].imshow(np.tanh(data[this_index, :, :, 0]))
                    e_img[row, col].set_title("prob: " + str(pred[evt_counter]))
                    e_img[row, col].annotate("eta = "+str(info[this_index,7]), xy=(0,0), xytext=(25,3), color='white')
                    e_img[row, col].annotate("phi = "+str(info[this_index,8]), xy=(0,0), xytext=(25,5), color='white')
                    evt_counter += 1
                this_type = nameStart.split("_")
                this_type = this_type[0]
                plt.savefig(saveDir + "failedImages_" + this_type + str(start_file) + ".png")
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
                    if j != 0 and j%5 == 0: row += 1
                    if j != 0 and j%25 == 0:
                       plt.savefig(saveDir + "failedImages_" + this_type + str(start_file) +"_"+str(counter)+ ".png")
                       counter += 1
                       row = 0
                    col = int(j%5)
                    e_img[row, col].imshow(np.tanh(data[this_index, :, :, 0]))
                    e_img[row, col].set_title("pred: " + str(pred[j]))
                    e_img[row, col].annotate("eta = "+str(info[this_index,7]), xy=(0,0), xytext=(25,3), color='white')
                    e_img[row, col].annotate("phi = "+str(info[this_index,8]), xy=(0,0), xytext=(25,5), color='white')
                plt.savefig(saveDir + "failedImages_" + this_type + str(start_file) +"_"+str(counter)+ ".png")
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

plotFile(file_e, index_e, pred_e, nameStartE, dataDir, imageDir, saveDir)
plotFile(file_b, index_b, pred_b, nameStartB, dataDir, imageDir, saveDir)



