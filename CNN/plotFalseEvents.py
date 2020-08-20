import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import utils


if len(sys.argv) > 1: filenameE = str(sys.argv[1])
else: print("need to specify electron input file")
if len(sys.argv) > 2: filenameB = str(sys.argv[2])
else:print("need to specify background input file")

imageDir = "/data/disappearingTracks/electron_selection_threshold_5gt0p5_V2/"
dataDir = "/home/mcarrigan/disTracksML/DisappTrksML/CNN/cnn_TestFull/outputFiles/"
nameStartE = "e_0p25_tanh_"
nameStartB = "bkg_0p25_tanh_"
tag = ""
saveDir = '/home/mcarrigan/disTracksML/plots/falseEvents/fullTest_5gt0p5/'

filenameE = dataDir + filenameE
filenameB = dataDir + filenameB

dtype = [('file', int), ('index', int), ('truth', int), ('prediction', float)]


events_e = np.load(filenameE)
events_e = np.reshape(events_e, (-1,4))
events_e = events_e[np.argsort(events_e[:,0])]
file_e = events_e[:, 0]
index_e = events_e[:, 1]
truth_e = events_e[:, 2]
pred_e = events_e[:, 3]

events_b = np.load(filenameB)
events_b = np.reshape(events_b, (-1, 4))
events_b = events_b[np.argsort(events_b[:,0])]
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

def plotFile(files, index, pred, nameStart, dataDir, imageDir, saveDir):
    start_file = -1
    file_events = []
    evt_counter = 0
    eta = []
    phi = []
    for event in range(len(files)):
        if event == 0: start_file = int(files[event])
        if start_file == files[event]: file_events.append(index[event])
        if start_file != files[event] or event == len(files)-1:
            filename = nameStart + str(start_file) + tag + ".npz"
            data, info = utils.load_electron_data(imageDir, filename)
            data = data[:,1:]
            data = np.reshape(data, [len(data),40,40,4])
            info = np.reshape(info, (-1, 8))
            row = int(0)
            if len(file_events) <= 25 :
                plot_len = len(file_events) / 5 + 1
                plot_len = int(plot_len)
                if plot_len < 2: plot_len = 2
                fig2, e_img = plt.subplots(plot_len, 5, figsize = (25, plot_len*5))
                for j in range(len(file_events)):
                    eta.append(info[int(file_events[j]),6])
                    phi.append(info[int(file_events[j]),7])
                    if j != 0 and j%5 == 0: row += 1
                    col = int(j%5)
                    print("event num ", file_events[j], " index ", j, " data size ", data.shape)
                    e_img[row, col].imshow(data[int(file_events[j]), :, :, 0])
                    e_img[row, col].set_title("prob: " + str(pred[evt_counter]))
                    e_img[row, col].annotate("eta = "+str(info[int(file_events[j],6)]), xy=(0,0), xytext=(35,3), color='white')
                    e_img[row, col].annotate("phi = "+str(info[int(file_events[j],7)]), xy=(0,0), xytext=(35,8), color='white')
                    print("Event counter ", evt_counter,  " pred ", pred[evt_counter]) 
                    evt_counter += 1
                this_type = nameStart.split("_")
                this_type = this_type[0]
                plt.savefig(saveDir + "failedImages_" + this_type + str(start_file) + ".png")
                start_file = int(files[event])
                file_events = [index[event]]

            else:
                this_type = nameStart.split("_")
                this_type = this_type[0]
                counter = 0
                fig2, e_img = plt.subplots(5, 5, figsize = (25, 25))
                for j in range(len(file_events)):
                    eta.append(info[int(file_events[j]),6])
                    phi.append(info[int(file_events[j]),7])
                    if j != 0 and j%5 == 0: row += 1
                    if j != 0 and j%25 == 0:
                       plt.savefig(saveDir + "failedImages_" + this_type + str(start_file) +"_"+str(counter)+ ".png")
                       counter += 1
                       row = 0
                    col = int(j%5)
                    print("counter ", j, " row: ", row, " col: ", col)
                    e_img[row, col].imshow(data[int(file_events[j]), :, :, 0])
                    e_img[row, col].set_title("pred: " + str(pred[j]))
                    e_img[row, col].annotate("eta = "+str(info[int(file_events[j],6)]), xy=(0,0), xytext=(35,3), color='white')
                    e_img[row, col].annotate("phi = "+str(info[int(file_events[j],7)]), xy=(0,0), xytext=(35,8), color='white')
                plt.savefig(saveDir + "failedImages_" + this_type + str(start_file) +"_"+str(counter)+ ".png")
                start_file = int(files[event])
                file_events = [index[event]]
    fig3, h_etaPhi = plt.subplots(1,2, figsize=(5,10))
    h_etaPhi[0].hist(eta, bins = 40)
    h_etaPhi[0].set_title("Eta of Misidentified Events")
    h_etaPhi[0].set(xlabel='Eta')
    h_etaPhi[1].hist(phi, bins = 40)             
    h_etaPhi[1].set_title("Phi of Misidentified Events")
    h_etaPhi[1].set(xlabel='Phi')
    plt.savefig(saveDir + "failedImages_EtaPhi.png")


plotFile(file_e, index_e, pred_e, nameStartE, dataDir, imageDir, saveDir)
plotFile(file_b, index_b, pred_b, nameStartB, dataDir, imageDir, saveDir)



