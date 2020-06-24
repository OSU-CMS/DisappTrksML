import numpy as np

def getBatch(batch_size, file_num, unused_events, imgDir, classDir):
    this_batch = []
    classes = []
    event_count = 0
    file_count = len(file_num)
    while event_count < batch_size-1:
        #choose random file
        rand_file = np.random.randint(0, file_count)
        print("Loading file...", file_num[rand_file])
        print("Event count...", event_count)
        #load file
        this_data = np.load(imgDir + file_num[rand_file])
        class_num = file_num[rand_file].split('_')
        class_num = class_num[2]
        this_classes = np.load(classDir + 'classes_0p25' + class_num + '_Clean.npy')
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
                classes = np.reshape(this_classes[choose[i]], (-1, 4))
            #add event to batch
            this_img = np.reshape(this_data[choose[i]], (-1, 40, 40, 3))
            this_class = np.reshape(this_classes[choose[i]], (-1, 4))
            this_batch = np.concatenate((this_batch, this_img))
            classes = np.concatenate((classes, this_class))
        #remove events from unused events
        event_count += num
        eventsRe = np.setdiff1d(eventsRe, choose)
        unused_events[rand_file] = eventsRe
        #print("Remaining events...")
        #print(eventsRe[:10])
        #print("Chosen events")
        #print(choose)
        #print("Removed used events")
    return this_batch, classes, unused_events
