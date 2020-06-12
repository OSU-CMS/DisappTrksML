import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from functools import partial
from itertools import repeat
import numpy as np

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


def nested_defaultdict(default_factory, depth=1):
    result = partial(defaultdict, default_factory)
    for _ in repeat(None, depth - 1):
        result = partial(defaultdict, result)
    return result()

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