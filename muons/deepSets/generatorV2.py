import numpy as np
from tensorflow import keras
from collections import Counter
import sys

class BalancedGenerator(keras.utils.Sequence):

	def __init__(self, 
				 file_ids, input_dir='', 
				 batch_size=32, 
				 batch_ratio=0.5,
				 shuffle=True,
				 with_info=False,
				 maxHits=100,
				 maxHits_calos=100):
		self.file_ids = file_ids
		self.input_dir = input_dir
		self.batch_size = batch_size
		self.batch_ratio = batch_ratio
		self.num_signal_batch = int(np.ceil(self.batch_ratio * self.batch_size))
		self.num_background_batch = self.batch_size - self.num_signal_batch
		self.shuffle = shuffle
		self.with_info = with_info
		self.maxHits = maxHits
		self.maxHits_calos = maxHits_calos

		self.signal_files = np.array([])
		self.background_files = np.array([])
		self.signal_events = np.array([])
		self.background_events = np.array([])
		self.num_signal = 0
		self.num_background = 0

		self.create_event_file_lists()
		self.on_epoch_end()

	def create_event_file_lists(self):

		for idx in self.file_ids:
			fin = np.load(self.input_dir + 'images_' + idx + '.root.npz')
			self.signal_files = np.concatenate((self.signal_files,np.ones(fin['signal'].shape[0])*int(idx)))
			self.signal_events = np.concatenate((self.signal_events,np.arange(fin['signal'].shape[0],dtype=int)))
			self.background_files = np.concatenate((self.background_files,np.ones(fin['background'].shape[0])*int(idx)))
			self.background_events = np.concatenate((self.background_events,np.arange(fin['background'].shape[0],dtype=int)))

		self.num_signal = len(self.signal_events)
		self.num_background = len(self.background_events)

		print "Found", self.num_signal + self.num_background,"events:", self.num_signal,"signal events and",self.num_background,"background events"

	def __len__(self):
		max_signal_batches = np.floor(self.num_signal / self.num_signal_batch)
		max_background_batches = np.floor(self.num_background / self.num_background_batch)
		return int(min(max_signal_batches, max_background_batches))

	def __getitem__(self, index):
		return self.__data_generation(index)

	def on_epoch_end(self):
		if self.shuffle:
			events = np.array([])
			files = np.array([])
			counter = Counter(self.signal_files)
			files_set = np.array(list(counter.keys()))
			p = np.random.permutation(len(files_set))
			files_set = files_set[p]
			for file in files_set:
				events = np.concatenate((events, np.arange(counter[file])))
				files = np.concatenate((files, np.ones(counter[file])*file))
			self.signal_events = events
			self.signal_files = files

			events = np.array([])
			files = np.array([])
			counter = Counter(self.background_files)
			files_set = np.array(list(counter.keys()))
			p = np.random.permutation(len(files_set))
			files_set = files_set[p]
			for file in files_set:
				events = np.concatenate((events, np.arange(counter[file])))
				files = np.concatenate((files, np.ones(counter[file])*file))
			self.background_events = events
			self.background_files = files

	def __data_generation(self, index):

		X, X_calos, X_info = None, None, None
		X_files = self.signal_files[index * self.num_signal_batch : (index + 1) * self.num_signal_batch].astype(int)
		X_events = self.signal_events[index * self.num_signal_batch : (index + 1) * self.num_signal_batch].astype(int)
		files = list(set(X_files))
		for file in files:
			events_this_file = X_events[np.where(X_files == file)]
			if X is None:
				X = np.load(self.input_dir + 'images_' + str(int(file)) + '.root.npz', allow_pickle=True)['signal'][events_this_file]
				X_calos = np.load(self.input_dir + 'images_' + str(int(file)) + '.root.npz', allow_pickle=True)['signal_calos'][events_this_file]
				if(self.with_info): X_info = np.load(self.input_dir + 'images_' + str(int(file)) + '.root.npz', allow_pickle=True)['signal_info'][events_this_file]
			else:
				X = np.vstack((X,np.load(self.input_dir + 'images_' + str(int(file)) + '.root.npz', allow_pickle=True)['signal'][events_this_file]))
				X_calos = np.vstack((X_calos,np.load(self.input_dir + 'images_' + str(int(file)) + '.root.npz', allow_pickle=True)['signal_calos'][events_this_file]))
				if(self.with_info): X_info = np.vstack((X_info,np.load(self.input_dir + 'images_' + str(int(file)) + '.root.npz', allow_pickle=True)['signal_info'][events_this_file]))

		if X is None: sys.exit("X is None")

		X_files = self.background_files[index * self.num_background_batch : (index + 1) * self.num_background_batch].astype(int)
		X_events = self.background_events[index * self.num_background_batch : (index + 1) * self.num_background_batch].astype(int)
		files = list(set(X_files))
		for file in files:
			events_this_file = X_events[np.where(X_files == file)]
			X = np.vstack((X,np.load(self.input_dir + 'images_' + str(int(file)) + '.root.npz', allow_pickle=True)['background'][events_this_file]))
			X_calos = np.vstack((X_calos,np.load(self.input_dir + 'images_' + str(int(file)) + '.root.npz', allow_pickle=True)['background_calos'][events_this_file]))
			if(self.with_info): X_info = np.vstack((X_info,np.load(self.input_dir + 'images_' + str(int(file)) + '.root.npz', allow_pickle=True)['background_info'][events_this_file]))

		y = np.concatenate((np.ones(self.num_signal_batch), np.zeros(self.num_background_batch)))

		p = np.random.permutation(len(X))
		X = X[p]
		X_calos = X_calos[p]
		y = y[p]
		if(self.with_info):
			X_info = X_info[p]
			X_info = X_info[:,[4,8,9,13,14,15,16]]

		X = X[:,:self.maxHits,:]
		X_calos =  X_calos[:,:self.maxHits,:]

		if self.with_info:
			return [X,X_calos,X_info], keras.utils.to_categorical(y, num_classes=2)
		return [X,X_calos], keras.utils.to_categorical(y, num_classes=2)


class Generator(keras.utils.Sequence):

	def __init__(self, 
				 file_ids, input_dir='', 
				 batch_size=32, 
				 with_info=False,
				 maxHits=100,
				 maxHits_calos=100):
		self.file_ids = file_ids
		self.input_dir = input_dir
		self.batch_size = batch_size
		self.with_info = with_info
		self.maxHits = maxHits
		self.maxHits_calos = maxHits_calos

		self.files = np.array([])
		self.events = np.array([])
		self.classes = np.array([])

		self.create_event_file_lists()

	def create_event_file_lists(self):

		for idx in self.file_ids:
			fin = np.load(self.input_dir + 'images_' + idx + '.root.npz')
			num_signal_file = int(fin['signal'].shape[0])
			self.files = np.concatenate((self.files,np.ones(num_signal_file)*int(idx)))
			self.events = np.concatenate((self.events,np.arange(num_signal_file,dtype=int)))
			self.classes = np.concatenate((self.classes,np.ones(num_signal_file)))

			num_bkg_file = int(fin['background'].shape[0])
			self.files = np.concatenate((self.files,np.ones(num_bkg_file)*int(idx)))
			self.events = np.concatenate((self.events,np.arange(num_bkg_file,dtype=int)))
			self.classes = np.concatenate((self.classes,np.zeros(num_bkg_file)))

		assert len(self.files) == len(self.events) and len(self.files) == len(self.classes)
 		print "Found",len(self.files),"events:", len(self.classes[np.where(self.classes==1)]),"signal events and",len(self.classes[np.where(self.classes==0)]),"background events"

 		self.classes = self.classes.astype(int)
 		self.files = self.files.astype(int)
 		self.events = self.events.astype(int)

	def __len__(self):
		return int(np.floor(len(self.files) / self.batch_size))

	def __getitem__(self, index):
		return self.__data_generation(index)

	def __data_generation(self, index):

		class_labels = ['background','signal']
		X, X_calos, X_info = None, None, None
		X_files = self.files[index * self.batch_size : (index + 1) * self.batch_size]
		X_events = self.events[index * self.batch_size : (index + 1) * self.batch_size]
		X_classes = self.classes[index * self.batch_size : (index + 1) * self.batch_size]
		
		for file in list(set(X_files)):
			events_this_file = X_events[np.where(X_files == file)]
			classes_this_file = X_classes[np.where(X_files == file)]

			for c in list(set(classes_this_file)):
				events_this_class = events_this_file[np.where(classes_this_file==c)]
				if X is None:
					X = np.load(self.input_dir + 'images_' + str(int(file)) + '.root.npz', allow_pickle=True)[class_labels[c]][events_this_class]
					X_calos = np.load(self.input_dir + 'images_' + str(int(file)) + '.root.npz', allow_pickle=True)[class_labels[c]+'_calos'][events_this_class]
					X_info = np.load(self.input_dir + 'images_' + str(int(file)) + '.root.npz', allow_pickle=True)[class_labels[c]+'_info'][events_this_class]
				else:
					X = np.vstack((X,np.load(self.input_dir + 'images_' + str(int(file)) + '.root.npz', allow_pickle=True)[class_labels[c]][events_this_class]))
					X_calos = np.vstack((X_calos,np.load(self.input_dir + 'images_' + str(int(file)) + '.root.npz', allow_pickle=True)[class_labels[c]+'_calos'][events_this_class]))
					X_info = np.vstack((X_info,np.load(self.input_dir + 'images_' + str(int(file)) + '.root.npz', allow_pickle=True)[class_labels[c]+'_info'][events_this_class]))

		y = X_info[:,3]

		p = np.random.permutation(len(X))
		X = X[p]
		X_calos = X_calos[p]
		y = y[p]
		if(self.with_info):
			X_info = X_info[p]
			X_info = X_info[:,[4,8,9,13,14,15,16]]

		X = X[:,:self.maxHits,:]

		if self.with_info:
			return [X,X_calos,X_info], keras.utils.to_categorical(y, num_classes=2)
		return [X,X_calos], keras.utils.to_categorical(y, num_classes=2)