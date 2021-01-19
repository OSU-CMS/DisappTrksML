import numpy as np
from tensorflow import keras
from collections import Counter
import sys
from sklearn.preprocessing import normalize

class DataGenerator(keras.utils.Sequence):

	def __init__(self, 
				 file_ids, input_dir='', 
				 batch_size=32, 
				 shuffle=True,
				 with_info=False,
				 maxHits=100,
				 flatten=False):
		self.file_ids = file_ids
		self.input_dir = input_dir
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.with_info = with_info
		self.maxHits = maxHits
		self.flatten = flatten

		self.files = np.array([])
		self.events = np.array([])

		self.nEvents = 0

		self.create_event_file_lists()
		self.on_epoch_end()

	def create_event_file_lists(self):

		for idx in self.file_ids:
			fin = np.load(self.input_dir + 'images_' + idx + '.root.npz')
			self.files = np.concatenate((self.files,np.ones(fin['background'].shape[0])*int(idx)))
			self.events = np.concatenate((self.events,np.arange(fin['background'].shape[0],dtype=int)))

		self.nEvents = len(self.events)
		print self.nEvents

	def __len__(self):
		return np.floor(self.nEvents / self.batch_size)

	def __getitem__(self, index):
		return self.__data_generation(index)

	def on_epoch_end(self):
		if self.shuffle:
			events = np.array([])
			files = np.array([])
			counter = Counter(self.files)
			files_set = np.array(list(counter.keys()))
			p = np.random.permutation(len(files_set))
			files_set = files_set[p]
			for file in files_set:
				events = np.concatenate((events, np.arange(counter[file])))
				files = np.concatenate((files, np.ones(counter[file])*file))
			self.events = events
			self.files = files

	def __data_generation(self, index):

		X, X_info = None, None
		X_files = self.files[index * self.batch_size : (index + 1) * self.batch_size].astype(int)
		X_events = self.events[index * self.batch_size : (index + 1) * self.batch_size].astype(int)
		files = list(set(X_files))
		for file in files:
			events_this_file = X_events[np.where(X_files == file)]
			if X is None:
				X = np.load(self.input_dir + 'images_' + str(int(file)) + '.root.npz',allow_pickle=True)['background'][events_this_file]
				X_info = np.load(self.input_dir + 'images_' + str(int(file)) + '.root.npz',allow_pickle=True)['background_info'][events_this_file]
			else:
				X = np.vstack((X,np.load(self.input_dir + 'images_' + str(int(file)) + '.root.npz',allow_pickle=True)['background'][events_this_file]))
				X_info = np.vstack((X_info,np.load(self.input_dir + 'images_' + str(int(file)) + '.root.npz',allow_pickle=True)['background_info'][events_this_file]))

		if X is None: sys.exit("X is None")

		p = np.random.permutation(len(X))
		X = X[p]

		X = X[:,:self.maxHits,:]

		if(self.flatten):

			#for i in range(X.shape[-1]): X[:,:,i] = normalize(X[:,:,i])

			X = np.reshape(X,(self.batch_size,self.maxHits*4))

		return X, X