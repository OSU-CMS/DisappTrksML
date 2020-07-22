import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from functools import partial
from itertools import repeat
import numpy as np

def load_electron_data(dataDir, tag):
    data = np.load(dataDir+tag)
    print("Loaded",len(data['images']),"events from",dataDir,tag)
    return data['images'], data['infos']

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

def getBatch(batch_size, file_num, unused_events, imgDir):
    this_batch = []
    classes = []
    event_count = 0
    file_count = len(file_num)
    events_remaining = 0
    for i in range(len(unused_events)):
        if len(unused_events) > 0: events_remaining += len(unused_events[i])
    if batch_size >= events_remaining:
        for i in range(len(unused_events)):
            if len(unused_events) > 0:
                print("Loading file " + file_num[i] + " with " + str(len(unused_events[i])) + " events")
                images, info = utils.load_electron_data(imgDir, file_num[i])
                this_data = images[:,1:]
                this_data = np.reshape(this_data, [-1,40,40,4])
                this_data = this_data[:, :, :, [0, 2, 3]]
                this_classes = np.array([x[1] for x in info])
                this_classes = this_classes.astype(int)
                for j in range(len(unused_events[i])):
                    if j % 500 == 0: print("Loaded " + str(j) + " events")
                    if event_count == 0:
                        this_batch = this_data[unused_events[i][j]]
                        classes = [this_classes[unused_events[i][j]]]
                    else:
                        this_batch = np.concatenate((this_batch, this_data[unused_events[i][j]]))
                        classes.append(this_classes[unused_events[i][j]])
                    event_count += 1
        this_batch = np.reshape(this_batch, (-1, 40, 40, 3))
        unused_events = []
        return this_batch, classes, unused_events
    while event_count < batch_size-1:
        #choose random file
        rand_file = np.random.randint(0, file_count)
        print("Loading file...", file_num[rand_file])
        print("Event count...", event_count)
        #load file
        images, info = utils.load_electron_data(imgDir, file_num[rand_file])
        this_data = images[:,1:]
        this_data = np.reshape(this_data, [-1,40,40,4])
        this_data = this_data[:, :, :, [0, 2, 3]]
        this_classes = np.array([x[1] for x in info])
        this_classes = this_classes.astype(int)
        #get events that are remaining (unused)
        eventsRe = unused_events[rand_file]
        #if enough events in file and less than 10% of batch_size needed, fill batch
        eventRange = len(eventsRe)
        num = np.random.randint(0, eventRange)
        #print("num1", num)
        if len(eventsRe) > (batch_size - event_count):
            num = np.random.randint(0, batch_size-event_count)
            #print("num2", num)
            if float(float(batch_size - event_count)/batch_size) <= 0.1:
                num = int(batch_size - event_count)
                #print("num3", num)
                #print(float(float(batch_size-event_count)/batch_size))
        #random number of events to select from file
        print("Selecting ... " + str(num) + ' events')
        #choose events 
        choose = np.random.choice(eventsRe, num)
        #add these events to counter
        for i in range(len(choose)):
            if (event_count ==0 and i ==0): 
                this_batch = np.reshape(this_data[choose[i]], (-1, 40, 40, 3))
                #classes = np.reshape(this_classes[choose[i]], (-1, 1))
                classes = [this_classes[choose[i]]]
            #add event to batch
            this_img = np.reshape(this_data[choose[i]], (-1, 40, 40, 3))
            #this_class = np.reshape(this_classes[choose[i]], (-1, 1))
            this_class = np.array(this_classes[choose[i]])
            this_batch = np.concatenate((this_batch, this_img))
            classes.append(this_classes[choose[i]])
        #remove events from unused events
        event_count += num
        eventsRe = [x for x in eventsRe if x not in choose]
        #eventsRe = np.setdiff1d(eventsRe, choose)
        #for i in range(unused_events.shape[0]):
         #   if i != rand_file: usused_events
        #unused_events[rand_file] = np.resize(len(eventsRe))
        unused_events[rand_file] = eventsRe
        #print("Remaining events...")
        #print(eventsRe[:10])
        #print("Chosen events")
        #print(choose)
        #print("Removed used events")
    return this_batch, classes, unused_events


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
