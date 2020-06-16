import sys
import os

file_num = [100, 108, 111, 113, 1148, 12, 1134, 18, 190, 1, 1350, 15, 1732, 1751, 1796]

saveFile = sys.argv[1]

for i in range(len(file_num)):
    print("Working on file: " + str(i) + "/" + str(len(file_num)))
    if(i==0): os.system('python3 trainAutoencoder.py ' + str(file_num[i]) + ' ' + saveFile)
    os.system('python3 trainAutoencoder.py ' + str(file_num[i]) + ' ' + saveFile + ' ' + saveFile)
 
print("Done")

