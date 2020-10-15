import os
import numpy as np

import utils
#from generator import generator, load_data
#from model import buildModel, buildModelWithEventInfo

def run_batch_validation(model, weights, batchDir, dataDir, plotDir):
	print("------------------STARTING VALIDATION--------------------")
	model.load_weights(weights)

	# load the batches used to train and validate
	val_e_file_batches = np.load(batchDir+'e_files_valBatches.npy', allow_pickle=True)
	val_e_event_batches = np.load(batchDir+'e_events_valBatches.npy', allow_pickle=True)
	val_bkg_file_batches = np.load(batchDir+'bkg_files_valBatches.npy', allow_pickle=True)
	val_bkg_event_batches = np.load(batchDir+'bkg_events_valBatches.npy', allow_pickle=True)
	file_batches = np.concatenate((val_e_file_batches,val_bkg_file_batches))
	event_batches = np.concatenate((val_e_event_batches,val_bkg_event_batches))
	class_labels = np.concatenate((['e']*val_e_file_batches.shape[0],['bkg']*val_bkg_file_batches.shape[0]))

	predictions, infos = [],[]
	for events,files,class_label in zip(event_batches,file_batches,class_labels):
		events = load_data(files,events,class_label,dataDir)
		batch_infos = load_data(files,events,class_label+'_infos',dataDir)

		events = events[:,3:]
		events = np.reshape(events,(events.shape[0],100,4))

		preds = model.predict(events)
		predictions.append(preds)
		infos.append(batch_infos)

	utils.metrics(true[:,1], predictions[:,1], plotDir, threshold=0.5)

	print()
	print(utils.bcolors.GREEN+"Saved metrics to "+plotDir+utils.bcolors.ENDC)
	print()

	np.savez_compressed(batchDir+"validation_outputs",
						truth = true,
						predicted = predictions,
						indices = indices)


def run_batch_validation2(model, weights, batchDir, dataDir, plotDir, batch_size):
	print("------------------STARTING VALIDATION--------------------")
	model.load_weights(weights)

	# load the batches used to train and validate
	val_e_file_batches = np.load(batchDir+'e_files_valBatches.npy', allow_pickle=True)
	val_e_event_batches = np.load(batchDir+'e_events_valBatches.npy', allow_pickle=True)
	val_bkg_file_batches = np.load(batchDir+'bkg_files_valBatches.npy', allow_pickle=True)
	val_bkg_event_batches = np.load(batchDir+'bkg_events_valBatches.npy', allow_pickle=True)

	print("Define Generator")
	val_generator = generator(val_e_file_batches, val_bkg_file_batches, val_e_event_batches, val_bkg_event_batches, 
		batch_size, dataDir, True, False, True)
	print("Reset Generator")
	val_generator.reset()
	print("Get Predictions")
	predictions = model.predict(val_generator, verbose=2)
	true = val_generator.get_y_batches()
	print("Get Indices of Events")
	indices = val_generator.get_indices_batches()

	cm = np.zeros((2,2)) 
	for t,pred,index in zip(true,predictions, indices):
		if(pred[1]>0.5):
			if(t[1]>0.5): 
				cm[1][1]+=1
			else: 
				cm[1][0]+=1		
		else:
			if(t[1]>0.5): 
				cm[0][1]+=1
			else: cm[0][0]+=1
	print(cm)

	utils.metrics(true[:,1], predictions[:,1], plotDir, threshold=0.5)

	print()
	print(utils.bcolors.GREEN+"Saved metrics to "+plotDir+utils.bcolors.ENDC)
	print()

	np.savez_compressed(batchDir+"validation_outputs",
						truth = true,
						predicted = predictions,
						indices = indices)

def run_validation(model, weights, outDir, dataDir,plotDir=""):
	print("------------------STARTING VALIDATION--------------------")
	model.load_weights(weights)

	predictions, infos = np.array([]), np.array([])
	file_count = 0
	for file in os.listdir(dataDir):
		if(".npz" not in file): continue
		if("images_" not in file): continue
		#if(file_count > 5): break
		data = np.load(dataDir+file)
		events = data['images']
		if events.shape[0] == 0: continue
		events = events[:,5:]
		print("Loaded " + str(len(events)) + " from file " + str(file))
		indices = data['images'][:,:5]
		events = np.reshape(events,(-1,40,40,4))
		events = np.tanh(events[:, :, :, [0,2,3]])
		batch_infos = data['infos']
		preds = model.predict(events)
		if file_count == 0:
			predictions = np.array(preds)
			infos = np.array(batch_infos)
		else:
			predictions = np.concatenate((predictions, preds))
			infos = np.concatenate((infos, batch_infos))
		file_count += 1
	infos = np.reshape(infos, (len(infos),16)) 
	predictions = np.array(predictions)
	infos = np.concatenate((infos, predictions), axis=1)
	np.save(outDir + "validation_outputs.npy",infos)

if __name__ == "__main__":

	dataDir = "/store/user/llavezzo/disappearingTracks/converted_deepSets100_Zee_V3/"
	batchDir = "deepSets_4/outputFiles/"
	plotDir = "deepSets_4/plots/"
	weights = "deepSets_4/weights/lastEpoch.h5"

	model = buildModelWithEventInfo()

	model.compile(optimizer=optimizers.Adam(), 
				  loss='categorical_crossentropy', 
				  metrics=['accuracy'])

	run_batch_validation(model, weights, batchDir, dataDir, plotDir, 128)
	# run_validation(model,weights,dataDir,plotDir)


