import numpy as np
import pickle 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# loop through folders (expects metrics.pkl in each) and plot val_acc and val_loss
def print_kfold_metrics(folders):
	indices, val_losses, val_accs = [], [], []
	for folder in folders:
		with open(folder+'/metrics.pkl','rb') as f:
			metrics = pickle.load(f)
		indices.append(folder)
		val_losses.append(metrics['val_loss'])
		val_accs.append(metrics['val_acc'])

	plt.scatter(indices,val_losses,label="validation loss")
	plt.xlabel("Parameter Index")
	plt.legend()
	plt.xticks(rotation = 90)
	plt.subplots_adjust(bottom=0.3)
	plt.savefig("results_loss.png")
	plt.clf()

	plt.scatter(indices,val_accs,label="validation accuracy")
	plt.xlabel("Parameter Index")
	plt.legend()
	plt.xticks(rotation = 90)
	plt.subplots_adjust(bottom=0.3)
	plt.savefig("results_acc.png")

def plot_history(infile,outfile):

	with open(infile,'rb') as f:
		history = pickle.load(f)

	loss = history['loss']
	val_loss = history['val_loss']

	epochs = range(1, len(loss) + 1)

	plt.figure()
	plt.plot(epochs, loss, 'bo', label='Training loss')
	plt.plot(epochs, val_loss, 'b', label='Validation loss')
	plt.title('Training and validation loss')
	plt.legend()

	plt.savefig(outfile)


#####################

infiles = ['trainV2_param5/trainingHistory_'+str(i)+'.pkl' for i in range(5)]
outfiles = ['trainV2_param5/trainingHistory_'+str(i)+'.png' for i in range(5)]
for infile, outfile in zip(infiles, outfiles): 
	plot_history(infile,outfile)