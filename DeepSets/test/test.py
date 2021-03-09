import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

font_size = 14
plt.rcParams.update({'font.size': font_size})

def save_event(x,outf="event.png"):

		if(x.shape[0] == 404): x = x[4:]
		if(x.shape[0] == 400): x = np.reshape(x, (100,4))

		fig, ax = plt.subplots(1,1, figsize=(5,5))
				
		detType = 0
		im = x[np.where(x[:,3]==detType)]
		im = im[np.where(im[:,2]!=0)]
		#h = ax.hist2d(im[:,0],im[:,1],weights=im[:,2],range=[[-0.3,0.3],[-0.3,0.3]],cmap='cubehelix',bins=(80,80))
		h = ax.hist2d(im[:,0],im[:,1],weights=np.ones(len(im[:,2])),cmap='cubehelix',range=[[-0.3,0.3],[-0.3,0.3]],bins=(80,80))
		#fig.colorbar(h[3], ax = ax)
		ax.set_xlabel(r"$\Delta\eta_{(recHit-track)}$")
		ax.set_ylabel(r"$\Delta\phi_{(recHit-track)}$")

		ax.set_title("ECAL")

		fig.tight_layout()
		fig.savefig(outf, bbox_inches='tight')  
		plt.close(fig)


dataDir = "/store/user/llavezzo/disappearingTracks/images_DYJetsToLL_v5_converted/"
for fname in os.listdir(dataDir):
	if(".root" not in fname): continue
	infile = np.load(dataDir+fname,allow_pickle=True)
	signal = infile['signal']
	
	for i in range(signal.shape[0]):
		save_event(signal[i],"images/test"+str(i)+".pdf")
	break