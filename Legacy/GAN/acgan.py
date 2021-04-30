import numpy as np
import keras
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Embedding, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Model
from keras.optimizers import Adam
from keras.initializers import RandomNormal
import matplotlib.pyplot as plt
import numpy as np
import numpy.random
from numpy.random import choice
from numpy.random import randn
import os
import pickle
import utils
import random

def load_data(files, events, label, dataDir):
    lastFile = len(files)-1
    files.sort()
    for iFile, file in enumerate(files):
        if(file == -1): 
            images = np.array([])
            continue
        if(iFile == 0 and iFile != lastFile):
            images = np.load(dataDir+label+str(file)+'.npy')[events[0]:]

        elif(iFile == lastFile and iFile != 0):
            images = np.vstack((images,np.load(dataDir+label+str(file)+'.npy')[:events[1]+1]))

        elif(iFile == 0 and iFile == lastFile):
            images = np.load(dataDir+label+str(file)+'.npy')[events[0]:events[1]+1]

        elif(iFile != 0 and iFile != lastFile):
            images = np.vstack((images,np.load(dataDir+label+str(file)+'.npy')))
    return images

def build_discriminator(img_shape,n_classes=2):
    input = Input(img_shape)
    x = Conv2D(32*3, kernel_size=(4,4), strides=(2,2), padding="same")(input)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.25)(x)
    x = Conv2D(64*3, kernel_size=(4,4), strides=(2,2), padding="same")(x)
    x = ZeroPadding2D(padding=((0, 1), (0, 1)))(x)
    x = (LeakyReLU(alpha=0.2))(x)
    x = Dropout(0.25)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Conv2D(128*3, kernel_size=(4,4), strides=(2,2), padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.25)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Conv2D(256*3, kernel_size=(4,4), strides=(1,1), padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    # real/fake output
    out1 = Dense(1, activation='sigmoid')(x)
    # class label output
    out2 = Dense(n_classes, activation='softmax')(x)

    model = Model(input, [out1, out2])
    print("-- Discriminator -- ")
    model.summary()
    
    model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'], optimizer=Adam(lr=0.0002, beta_1=0.5))
    return model

# define the standalone generator model
def build_generator(latent_dim, n_classes=2):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # label input
    in_label = Input(shape=(1,))
    # embedding for categorical input
    li = Embedding(n_classes, 50)(in_label)
    # linear multiplication
    n_nodes = 10 * 10
    li = Dense(n_nodes, kernel_initializer=init)(li)
    # reshape to additional channel
    li = Reshape((10, 10, 1))(li)
    # image generator input
    in_lat = Input(shape=(latent_dim,))
    # foundation for 10x10 image
    n_nodes = 128 * 10 * 10
    gen = Dense(n_nodes, activation='relu',kernel_initializer=init)(in_lat)
    gen = Reshape((10, 10, 128))(gen)
    gen = BatchNormalization(momentum=0.8)(gen)
    # merge image gen and label input
    merge = Concatenate()([gen, li])

    #upsampling to 20x20
    x = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(merge)
    x = Activation("relu")(x)
    x = BatchNormalization(momentum=0.8)(x)
    #upsampling to 40x40
    x = Conv2DTranspose(64, (4,4),strides=(2,2), padding='same')(x)
    x = Activation("relu")(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Conv2D(3, kernel_size=3, padding="same")(x)
    
    # TESTING
    out = Activation("relu")(x)
    #out = Activation('tanh')(x)

    # define model
    model = Model([in_lat, in_label], out)
    print("-- Generator -- ")
    model.summary()
    return model

#generates and saves r random images
def save_imgs(generator, epoch, batch, r):
    noise = generate_latent_points(100,r)
    fake_classes = np.concatenate([np.ones(int(round(r/2))),np.zeros(r-int(round(r/2)))])
    assert len(fake_classes) == r, "length of fake classes must be equal to r"
    gen_imgs = generator.predict([noise,fake_classes])

    fig, axs = plt.subplots(r, 3)
    for i in range(r):
        for j in range(3):
            axs[i, j].imshow(gen_imgs[i, :, :, j], cmap='gray')
            if(fake_classes[i]==1):
                axs[i,0].set_title("e - ECAL", fontsize = 9)
                axs[i,1].set_title("e - HCAL", fontsize = 9)
                axs[i,2].set_title("e - Muon System", fontsize = 9)
            if(fake_classes[i]==0):
                axs[i,0].set_title("bkg - ECAL", fontsize = 9)
                axs[i,1].set_title("bkg - HCAL", fontsize = 9)
                axs[i,2].set_title("bkg - Muon System", fontsize = 9)
            axs[i, j].axis('off')
    plt.tight_layout()
    fig.savefig(plotDir + "acgan_%d_%d.png" % (epoch, batch))
    plt.close()
    
# define the combined generator and discriminator model, for updating the generator
def build_gan(g_model, d_model):
    # make weights in the discriminator not trainable
    d_model.trainable = False
    # connect the outputs of the generator to the inputs of the discriminator
    gan_output = d_model(g_model.output)
    # define gan model as taking noise and label and outputting real/fake and label outputs
    model = Model(g_model.input, gan_output)
    # compile model
    model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'], optimizer=Adam(lr=0.0002, beta_1=0.5))
    print("-- GAN -- ")
    model.summary()
    return model
    
def noisy_labels(y, p_flip):
    # determine the number of labels to flip
    n_select = int(p_flip * y.shape[0])
    # choose labels to flip
    flip_ix = choice([i for i in range(y.shape[0])], size=n_select)
    # invert the labels in place
    y[flip_ix] = 1 - y[flip_ix]
    return y

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape((n_samples, latent_dim))
    return x_input

def smooth_positive_labels(y):
    return y - 0.3 + (np.random.random(y.shape) * 0.5)

def smooth_negative_labels(y):
    return y + np.random.random(y.shape) * 0.3

if __name__ == "__main__":

    dataDir = "/data/disappearingTracks/electron_selection/"
    workDir = "acgan_results"
    plotDir = workDir + '/plots/'
    weightsDir = workDir + '/weights/'

    ############## config params ##############
    epochs=10
    batch_size=128
    save_interval=100
    latent_dim = 100
    nTotE = 15000
    nTotBkg = 15000
    ##########################################

    # create output directories
    os.system('mkdir '+str(workDir))
    os.system('mkdir '+str(plotDir))
    os.system('mkdir '+str(weightsDir))

    # import count dicts
    with open(dataDir+'eCounts.pkl', 'rb') as f:
        eCounts = pickle.load(f)
    with open(dataDir+'bkgCounts.pkl', 'rb') as f:
        bkgCounts = pickle.load(f)

    half_batch = int(batch_size / 2)

    # batches per epoch
    num_batches = int((nTotE + nTotBkg)*1.0/half_batch)

    # count how many e/bkg events in each batch
    ePerBatch = np.zeros(num_batches)
    iBatch = 0
    while np.sum(ePerBatch) < nTotE:
        ePerBatch[iBatch]+=1
        iBatch+=1
        if(iBatch == num_batches): iBatch = 0
    bkgPerBatch = np.asarray([batch_size-np.min(ePerBatch)]*num_batches)
    ePerBatch = ePerBatch.astype(int)
    bkgPerBatch = bkgPerBatch.astype(int)

    # fill lists of all events and files
    b_events, b_files = [], []
    for file, nEvents in bkgCounts.items():
        for evt in range(nEvents):
            b_events.append(evt)
            b_files.append(file)
    e_events, e_files = [], []
    for file, nEvents in eCounts.items():
        for evt in range(nEvents):
            e_events.append(evt)
            e_files.append(file)

    # make batches
    bkg_event_batches, bkg_file_batches = utils.make_batches(b_events, b_files, bkgPerBatch, num_batches)
    e_event_batches, e_file_batches = utils.make_batches(e_events, e_files, ePerBatch, num_batches)
   
    
    # create the discriminator
    discriminator = build_discriminator(img_shape=(40,40,3),n_classes=2)
    # create the generator
    generator = build_generator(latent_dim)
    # create the gan
    gan_model = build_gan(generator, discriminator)

    num_examples = num_batches*batch_size

    print(utils.bcolors.YELLOW+'Training on:'+utils.bcolors.ENDC)
    print(utils.bcolors.GREEN+'Number of examples: '+utils.bcolors.ENDC, num_examples)
    print(utils.bcolors.GREEN+'Number of Batches: '+utils.bcolors.ENDC, num_batches)
    print(utils.bcolors.GREEN+'Number of epochs: '+utils.bcolors.ENDC, epochs)

    d_loss_array = []
    g_loss_array = []

    # gan_model.load_weights(weightsDir+'G_epoch{0}.h5'.format(100))
    # discriminator.load_weights(weightsDir+'D_epoch{0}.h5'.format(100))

    for epoch in range(epochs + 1):
        for batch in range(num_batches):

            # noise images for the batch
            noise = generate_latent_points(100,half_batch)
            fake_classes = np.random.randint(0,2,size=half_batch)
            fake_images = generator.predict([noise,fake_classes])
            fake_labels = np.zeros((half_batch, 1))

            # real images, classes for batch
            idx = np.random.randint(0, num_batches)
            e = load_data(e_file_batches[idx], e_event_batches[idx], 'e_0p25_', dataDir)
            bkg = load_data(bkg_file_batches[idx],  bkg_event_batches[idx], 'bkg_0p25_', dataDir)
            real_images = np.vstack((e,bkg))
            
            real_classes = np.concatenate((np.ones(len(e)),np.zeros(len(bkg))))
            
            indices = list(range(real_images.shape[0]))
            random.shuffle(indices)
            real_images = real_images[indices[:half_batch],1:]
            real_images = np.reshape(real_images,(half_batch,40,40,4))
            real_images = real_images[:,:,:,[0,2,3]]
            real_classes = real_classes[indices[:half_batch]]
            
            real_labels = np.ones((half_batch, 1))
            
            #smooth and noisy labels
            real_labels = noisy_labels(real_labels,0.05)
            real_labels = smooth_positive_labels(real_labels)
            fake_labels = smooth_negative_labels(fake_labels)

            # Train the discriminator (real classified as 1 and generated as 0)
            d_loss_real = discriminator.train_on_batch(real_images, [real_classes,real_labels])
            d_loss_fake = discriminator.train_on_batch(fake_images, [fake_classes,fake_labels])

            # Train the generator
            labels = np.ones((batch_size, 1))
            classes = np.random.randint(0, 2, batch_size)
            noise = generate_latent_points(100,batch_size)
            
            g_loss = gan_model.train_on_batch([noise,classes], [labels,classes])
            
            # Track the progress
            if(batch % save_interval == 0): 
                print('epoch %d batch %d, dr[%.3f,%.3f], df[%.3f,%.3f], g[%.3f,%.3f]' % 
                  (epoch, batch, d_loss_real[1],d_loss_real[2], 
                   d_loss_fake[1],d_loss_fake[2], g_loss[1],g_loss[2]))

                save_imgs(generator, epoch, batch, 4)

        gan_model.save_weights(weightsDir+'G_epoch{0}.h5'.format(epoch))
        discriminator.save_weights(weightsDir+'D_epoch{0}.h5'.format(epoch))
