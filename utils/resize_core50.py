import numpy as np 
import cv2
import os
import glob

#copy in data core50_128x128 into core50_128x128_old
#change home/jb/ into your correct folder
path_dataset = '/home/leonardolabs/data/core50_128x128_old/*'

list_folder = glob.glob(path_dataset)
print(list_folder, len(list_folder))
#import ipdb; ipdb.set_trace()
for folder in list_folder:
    list_folder_L1 = glob.glob(folder+'/*')
    #print(list_folder_L1)
    for l1 in list_folder_L1:
        print(l1)
        imgs = glob.glob(l1+'/*.png')
        for img_path in imgs:
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            resized = cv2.resize(img, (64,64), interpolation = cv2.INTER_AREA)
            img_path_changed = img_path.replace('_old','')
            cv2.imwrite(img_path_changed, resized)

