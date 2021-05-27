import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Dense, TimeDistributed, Masking, Input, Lambda, Activation, BatchNormalization
from keras.layers.merge import concatenate

def buildModel(input_shape=(100,4), phi_layers=[64, 64, 256], f_layers=[64, 64, 64]):
	inputs = Input(shape=(input_shape[-1],))

	# build phi network for each individual hit
	phi_network = Masking()(inputs)
	for layerSize in phi_layers[:-1]:
		phi_network = Dense(layerSize)(phi_network)
		phi_network = Activation('relu')(phi_network)
		phi_network = BatchNormalization()(phi_network)
	phi_network = Dense(phi_layers[-1])(phi_network)
	phi_network = Activation('linear')(phi_network)

	# build summed model for latent space
	unsummed_model = Model(inputs=inputs, outputs=phi_network)
	set_input = Input(shape=input_shape)
	phi_set = TimeDistributed(unsummed_model)(set_input)
	summed = Lambda(lambda x: tf.reduce_sum(x, axis=1))(phi_set)
	phi_model = Model(inputs=set_input, outputs=summed)

	# define F (rho) network evaluating in the latent space
	f_inputs = Input(shape=(phi_layers[-1],)) # plus any other track/event-wide variable
	f_network = Dense(f_layers[0])(f_inputs)
	f_network = Activation('relu')(f_network)
	for layerSize in f_layers[1:]:
		f_network = Dense(layerSize)(f_network)
		f_network = Activation('relu')(f_network)
	f_network = Dense(2)(f_network)
	f_outputs = Activation('softmax')(f_network)
	f_model = Model(inputs=f_inputs, outputs=f_outputs)

	# build the DeepSets architecture
	deepset_inputs = Input(shape=input_shape)
	latent_space = phi_model(deepset_inputs)
	deepset_outputs = f_model(latent_space)
	model = Model(inputs=deepset_inputs, outputs=deepset_outputs)

	print(model.summary())

	return model


def buildModelWithEventInfo(input_shape=(100,4), info_shape=6, phi_layers=[64, 64, 256], f_layers=[64, 64, 64]):
	inputs = Input(shape=(input_shape[-1],))

	# build phi network for each individual hit
	phi_network = Masking()(inputs)
	for layerSize in phi_layers[:-1]:
		phi_network = Dense(layerSize)(phi_network)
		phi_network = Activation('relu')(phi_network)
		phi_network = BatchNormalization()(phi_network)
	phi_network = Dense(phi_layers[-1])(phi_network)
	phi_network = Activation('linear')(phi_network)

	# build summed model for latent space
	unsummed_model = Model(inputs=inputs, outputs=phi_network)
	set_input = Input(shape=input_shape)
	phi_set = TimeDistributed(unsummed_model)(set_input)
	summed = Lambda(lambda x: tf.reduce_sum(x, axis=1))(phi_set)
	phi_model = Model(inputs=set_input, outputs=summed)

	# define F (rho) network evaluating in the latent space
	f_inputs = Input(shape=(phi_layers[-1]+info_shape,)) # plus any other track/event-wide variable
	f_network = Dense(f_layers[0])(f_inputs)
	f_network = Activation('relu')(f_network)
	for layerSize in f_layers[1:]:
		f_network = Dense(layerSize)(f_network)
		f_network = Activation('relu')(f_network)
	f_network = Dense(2)(f_network)
	f_outputs = Activation('softmax')(f_network)
	f_model = Model(inputs=f_inputs, outputs=f_outputs)

	# build the DeepSets architecture
	deepset_inputs = Input(shape=input_shape)
	latent_space = phi_model(deepset_inputs)
	info_inputs = Input(shape=(info_shape,))
	deepset_inputs2 = concatenate([latent_space,info_inputs])
	deepset_outputs = f_model(deepset_inputs2)
	model = Model(inputs=[deepset_inputs,info_inputs], outputs=deepset_outputs)

	print(model.summary())

	return model