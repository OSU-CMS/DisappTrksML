import numpy as np
import pickle 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def print_kfold_metrics():
	indices, val_losses, val_accs = [], [], []
	for i in range(6):
		if i == 4: continue
		with open('train_param'+str(i)+'/metrics.pkl','rb') as f:
			metrics = pickle.load(f)
		indices.append(i)
		val_losses.append(metrics['val_loss'])
		val_accs.append(metrics['val_acc'])

	plt.scatter(indices,val_losses,label="validation loss")
	plt.xlabel("Parameter Index")
	plt.legend()
	plt.savefig("results_loss.png")
	plt.clf()

	plt.scatter(indices,val_accs,label="validation accuracy")
	plt.xlabel("Parameter Index")
	plt.legend()
	plt.savefig("results_acc.png")

def display_history(infile,outfile):

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

for i in range(3):
	display_history('trainV2_param5/trainingHistory_'+str(i)+'.pkl','trainV2_param5/trainingHistory_'+str(i)+'.png')