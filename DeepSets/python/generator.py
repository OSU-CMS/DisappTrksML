import numpy as np
from tensorflow import keras
from collections import Counter
import sys

class DataGenerator(keras.utils.Sequence):

	def __init__(self, 
				 file_ids, input_dir='.', 
				 batch_size=32, 
				 dim=(100,4), 
				 n_classes=2, 
				 shuffle=True):
		self.file_ids = file_ids
		self.input_dir = input_dir
		self.batch_size = batch_size
		self.dim = dim
		self.n_classes = n_classes
		self.shuffle = shuffle

		self.signal_data = None
		self.background_data = None
		self.num_signal = 0
		self.num_background = 0

		self.load_data()
		self.on_epoch_end()

	def load_data(self):
		for idx in self.file_ids:
			fin = np.load(self.input_dir + 'images_' + idx + '.root.npz')
			print(idx)
			if self.signal_data is None and self.background_data is None:
				self.signal_data = fin['signal']
				self.background_data = fin['background']
			else:
				self.signal_data = np.vstack((self.signal_data, fin['signal']))
				self.background_data = np.vstack((self.background_data, fin['background']))
		self.num_signal = self.signal_data.shape[0]
		self.num_background = self.background_data.shape[0]

	def __len__(self):
		max_signal_batches = np.floor(self.num_signal / self.batch_size)
		max_background_batches = np.floor(self.num_background / self.batch_size)
		return int(min(max_signal_batches, max_background_batches))

	def __getitem__(self, index):
		return self.__data_generation(index)

	def on_epoch_end(self):
		if self.shuffle:
			np.random.shuffle(self.signal_data)
			np.random.shuffle(self.background_data)

	def __data_generation(self, index):
		X = self.signal_data[index * self.batch_size : (index + 1) * self.batch_size]
		y = np.ones(self.batch_size)

		X = np.vstack((X, self.background_data[index * self.batch_size : (index + 1) * self.batch_size]))
		y = np.concatenate((y, np.zeros(self.batch_size)))

		p = np.random.permutation(len(X))
		X = X[p]
		y = y[p]

		return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
"""
class DataGeneratorV2(keras.utils.Sequence):

	def __init__(self, 
				 file_ids, input_dir='.', 
				 batch_size=32, 
				 dim=(100,4), 
				 n_classes=2, 
				 shuffle=True):
		self.file_ids = file_ids
		self.input_dir = input_dir
		self.batch_size = batch_size
		self.dim = dim
		self.n_classes = n_classes
		self.shuffle = shuffle

		self.file_ids_used = np.zeros(())

		self.on_epoch_end()

	def on_epoch_end(self):
		self.signal_file_ids = np.arange(len(self.file_ids))
		self.background_file_ids = np.arange(len(self.file_ids))
		if self.shuffle:
			np.random.shuffle(self.signal_file_ids)
			np.random.shuffle(self.background_file_ids)

	def load_data(self):
		for idx in self.file_ids:
			fin = np.load(self.input_dir + 'images_' + idx + '.root.npz')
			if self.signal_data is None and self.background_data is None:
				self.signal_data = fin['signal']
				self.background_data = fin['background']
			else:
				self.signal_data = np.vstack((self.signal_data, fin['signal']))
				self.background_data = np.vstack((self.background_data, fin['background']))
		self.num_signal = self.signal_data.shape[0]
		self.num_background = self.background_data.shape[0]

		print 'Ay got:'
		print '\t', self.num_signal, 'signal events'
		print '\t', self.num_background, 'background events'
		print 'shapes: signal_data =', self.signal_data.shape
		print '        background_data =', self.background_data.shape

	def __len__(self):
		max_signal_batches = np.floor(self.num_signal / self.batch_size)
		max_background_batches = np.floor(self.num_background / self.batch_size)
		return int(min(max_signal_batches, max_background_batches))

	def __getitem__(self, index):
		return self.__data_generation(index)

	def __data_generation(self, index):
		X = self.signal_data[index * self.batch_size : (index + 1) * self.batch_size]
		y = np.ones(self.batch_size)

		X = np.vstack((X, self.background_data[index * self.batch_size : (index + 1) * self.batch_size]))
		y = np.concatenate((y, np.zeros(self.batch_size)))

		p = np.random.permutation(len(X))
		X = X[p]
		y = y[p]

		return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
"""

class DataGeneratorV3(keras.utils.Sequence):

	def __init__(self, 
				 file_ids, input_dir='', 
				 batch_size=32, 
				 batch_ratio=0.5,
				 shuffle=True,
				 with_info=False,
				 maxHits=100):
		self.file_ids = file_ids
		self.input_dir = input_dir
		self.batch_size = batch_size
		self.batch_ratio = batch_ratio
		self.num_signal_batch = int(np.ceil(self.batch_ratio * self.batch_size))
		self.num_background_batch = self.batch_size - self.num_signal_batch
		self.shuffle = shuffle
		self.with_info = with_info
		self.maxHits = maxHits

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

		print self.num_signal
		print self.num_background

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

		X, X_info = None, None
		X_files = self.signal_files[index * self.num_signal_batch : (index + 1) * self.num_signal_batch].astype(int)
		X_events = self.signal_events[index * self.num_signal_batch : (index + 1) * self.num_signal_batch].astype(int)
		files = list(set(X_files))
		for file in files:
			events_this_file = X_events[np.where(X_files == file)]
			if X is None:
				X = np.load(self.input_dir + 'images_' + str(int(file)) + '.root.npz')['signal'][events_this_file]
				X_info = np.load(self.input_dir + 'images_' + str(int(file)) + '.root.npz')['signal_info'][events_this_file]
			else:
				X = np.vstack((X,np.load(self.input_dir + 'images_' + str(int(file)) + '.root.npz')['signal'][events_this_file]))
				X_info = np.vstack((X_info,np.load(self.input_dir + 'images_' + str(int(file)) + '.root.npz')['signal_info'][events_this_file]))

		if X is None: sys.exit("X is None")

		X_files = self.background_files[index * self.num_background_batch : (index + 1) * self.num_background_batch].astype(int)
		X_events = self.background_events[index * self.num_background_batch : (index + 1) * self.num_background_batch].astype(int)
		files = list(set(X_files))
		for file in files:
			events_this_file = X_events[np.where(X_files == file)]
			X = np.vstack((X,np.load(self.input_dir + 'images_' + str(int(file)) + '.root.npz')['background'][events_this_file]))
			X_info = np.vstack((X_info,np.load(self.input_dir + 'images_' + str(int(file)) + '.root.npz')['background_info'][events_this_file]))

		y = np.concatenate((np.ones(self.num_signal_batch), np.zeros(self.num_background_batch)))

		p = np.random.permutation(len(X))
		X = X[p]
		y = y[p]
		X_info = X_info[p]
		X_info = X_info[:,[4,8,9,12]]

		X = X[:,:self.maxHits,:]

		if self.with_info:
			return [X,X_info], keras.utils.to_categorical(y, num_classes=2)
		return X, keras.utils.to_categorical(y, num_classes=2)


class DataGeneratorV4(keras.utils.Sequence):

	def __init__(self, 
				 file_ids, input_dir='', 
				 batch_size=32, 
				 with_info=False,
				 maxHits=100):
		self.file_ids = file_ids
		self.input_dir = input_dir
		self.batch_size = batch_size
		self.with_info = with_info
		self.maxHits = maxHits

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
		X, X_info = None, None
		X_files = self.files[index * self.batch_size : (index + 1) * self.batch_size]
		X_events = self.events[index * self.batch_size : (index + 1) * self.batch_size]
		X_classes = self.classes[index * self.batch_size : (index + 1) * self.batch_size]
		
		for file in list(set(X_files)):
			events_this_file = X_events[np.where(X_files == file)]
			classes_this_file = X_classes[np.where(X_files == file)]

			for c in list(set(classes_this_file)):
				events_this_class = events_this_file[np.where(classes_this_file==c)]
				if X is None:
					X = np.load(self.input_dir + 'images_' + str(int(file)) + '.root.npz')[class_labels[c]][events_this_class]
					X_info = np.load(self.input_dir + 'images_' + str(int(file)) + '.root.npz')[class_labels[c]+'_info'][events_this_class]
				else:
					X = np.vstack((X,np.load(self.input_dir + 'images_' + str(int(file)) + '.root.npz')[class_labels[c]][events_this_class]))
					X_info = np.vstack((X_info,np.load(self.input_dir + 'images_' + str(int(file)) + '.root.npz')[class_labels[c]+'_info'][events_this_class]))

		y = X_info[:,3]

		p = np.random.permutation(len(X))
		X = X[p]
		y = y[p]
		X_info = X_info[p]
		X_info = X_info[:,[4,8,9,12]]

		X = X[:,:self.maxHits,:]

		if self.with_info:
			return [X,X_info], keras.utils.to_categorical(y, num_classes=2)
		return X, keras.utils.to_categorical(y, num_classes=2)