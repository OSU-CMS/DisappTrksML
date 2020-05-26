import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def save_event(x, dir, fname):
    
    fig, axs = plt.subplots(1,5,figsize=(10,10))
    
    for i in range(5):
        axs[i].imshow(x[:,:,i],cmap='gray')
    
    axs[0].set_title("None")
    axs[1].set_title("EB")
    axs[2].set_title("EE")
    axs[3].set_title("HCAL")
    axs[4].set_title("Muon")
    
    plt.savefig(dir+fname)

def plot_event(x):
    
    fig, axs = plt.subplots(1,5,figsize=(10,10))
    
    for i in range(5):
        axs[i].imshow(x[:,:,i],cmap='gray')
    
    axs[0].set_title("None")
    axs[1].set_title("EB")
    axs[2].set_title("EE")
    axs[3].set_title("HCAL")
    axs[4].set_title("Muon")
    
    plt.show()