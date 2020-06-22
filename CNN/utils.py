import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from functools import partial
from itertools import repeat
import numpy as np
import os
from collections import Counter
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

def save_event(x, dir, fname):
    
    fig, axs = plt.subplots(1,3,figsize=(10,10))
    
    for i in range(3):
        axs[i].imshow(x[:,:,i],cmap='gray')
    
    axs[0].set_title("ECAL")
    axs[1].set_title("HCAL")
    axs[2].set_title("Muon")
    
    plt.savefig(dir+fname)

def plot_event(x):
    
    fig, axs = plt.subplots(1,3,figsize=(10,10))
    
    for i in range(4):
        axs[i].imshow(x[:,:,i],cmap='gray')
    
    axs[0].set_title("ECAL")
    axs[1].set_title("HCAL")
    axs[2].set_title("Muon")
    
    plt.show()

# load the electron selected data
def load_electron_data(dataDir, tag):
    data = np.load(dataDir+'electron_selection'+tag+'.npz')
    print("Loaded",len(data['images']),"events from",dataDir,tag)
    return data['images'], data['infos']

# load data with the images, infos .npz structure
def load_all_data(dataDir, tag):

  full = []
  infos = []

  for filename in os.listdir(dataDir):
    if('.npz' in filename and tag in filename and 'images' in filename):
      temp = np.load(dataDir+filename)
      full.append(temp['images'])
      infos.append(temp['infos'])

  full = np.vstack(full)
  infos = np.vstack(infos)
  assert full.shape[0] == infos.shape[0], "Full images and infos are of different sizes"

  print("Loaded",full.shape[0],"events from",dataDir,tag)
  return full, infos

def nested_defaultdict(default_factory, depth=1):
    result = partial(defaultdict, default_factory)
    for _ in repeat(None, depth - 1):
        result = partial(defaultdict, result)
    return result()

def apply_oversampling(x_train, y_train, oversample_val=0.1):
  
  counter = Counter(y_train)
  print("Before",counter)
  c1 = x_train.shape[1]
  c2 = x_train.shape[2]
  c3 = x_train.shape[3]
  x_train = np.reshape(x_train,[x_train.shape[0],c1*c2*c3])

  print("Applying oversampling with value",oversample_val)
  oversample = RandomOverSampler(sampling_strategy=oversample_val)
  x_train, y_train = oversample.fit_resample(x_train, y_train)
  x_train = np.reshape(x_train,[x_train.shape[0],c1,c2,c3])
  
  counter = Counter(y_train)
  print("After",counter)

  return x_train, y_train

def apply_undersampling(x_train, y_train, undersample_val=0.1):
 
  counter = Counter(y_train)
  print("Before",counter)
  c1 = x_train.shape[1]
  c2 = x_train.shape[2]
  c3 = x_train.shape[3]
  x_train = np.reshape(x_train,[x_train.shape[0],c1*c2*c3])
  
  print("Applying undersampling with value",undersample_val)
  undersample = RandomUnderSampler(sampling_strategy=undersample_val)
  x_train, y_train = undersample.fit_resample(x_train, y_train)
  x_train = np.reshape(x_train,[x_train.shape[0],c1,c2,c3])
  
  counter = Counter(y_train)
  print("After",counter)

  return x_train, y_train


def plot_history(history, plotDir, variables=['accuracy','loss']):
  for var in variables:
    plt.plot(history.history[var],label='train')
    plt.plot(history.history['val'+var],label='test')
    plt.title(var + ' History')
    plt.ylabel(var)
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(plotDir+var+'_history.png')
    plt.clf()
    
def calc_cm(y_test,predictions):
    confusion_matrix = nested_defaultdict(int,2)
    for true,pred in zip(y_test, predictions):
        t = np.argmax(true)
        p = np.argmax(pred)
        confusion_matrix[t][p] += 1
    return confusion_matrix

def plot_certainty(y_test,predictions,f):

    correct_certainty, notcorrect_certainty = [],[]
    for true,pred in zip(y_test, predictions):
        if np.argmax(true) == np.argmax(pred):
            correct_certainty.append(pred[np.argmax(pred)])
        else:
            notcorrect_certainty.append(pred[np.argmax(pred)])
    
    plt.hist(correct_certainty,alpha=0.5,label='Predicted Successfully',density=True)
    plt.hist(notcorrect_certainty,alpha=0.5,label='Predicted Unsuccessfully',density=True)
    plt.title("Certainty")
    plt.legend()
    plt.savefig(f)
    plt.clf()


def plot_confusion_matrix(confusion_matrix, target_names, f='cm.png', title='Confusion Matrix', cmap=plt.cm.Blues):
    
    #convert to array of floats
    cm = np.zeros([2,2])
    for i in range(2):
        for j in range(2):
            cm[i][j] = confusion_matrix[i][j]
    cm = cm.astype(float)

    plt.imshow(cm, interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()

    width, height = cm.shape

    for x in range(width):
        for y in range(height):
            plt.annotate(str(cm[x][y]), xy=(y, x), 
                        horizontalalignment='center',
                        verticalalignment='center')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f, bbox_inches='tight')
    plt.clf()

def calc_binary_metrics(confusion_matrix):
    c1=1
    c2=0
    TP = confusion_matrix[c1][c1]
    FP = confusion_matrix[c2][c1]
    FN = confusion_matrix[c1][c2]
    TN = confusion_matrix[c2][c2]

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    return precision, recall

def plot_grid(gs, x_label, y_label, x_target_names, y_target_names, title = 'Grid Search', f='gs.png', cmap=plt.get_cmap('Blues')):
 
    #convert to array of floats
    grid = np.zeros([len(y_target_names),len(x_target_names)])
    for i in range(len(y_target_names)):
        for j in range(len(x_target_names)):
            grid[i][j] = round(gs[i][j],3)
    grid = grid.astype(float)

    plt.imshow(grid, interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks_x = np.arange(len(x_target_names))
    tick_marks_y = np.arange(len(y_target_names))
    plt.xticks(tick_marks_x, x_target_names)
    plt.yticks(tick_marks_y, y_target_names)
    
    plt.tight_layout()

    width, height = grid.shape

    for x in range(width):
        for y in range(height):
            plt.annotate(str(grid[x][y]), xy=(y, x), 
                        horizontalalignment='center',
                        verticalalignment='center')

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(f, bbox_inches='tight')
    plt.clf()