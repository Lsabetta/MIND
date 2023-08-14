
import cv2
import random
import glob
import numpy as np
from torch.utils.data import Dataset
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union
import os 

def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folders in a dataset.

    See :class:`DatasetFolder` for details.
    """
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


class tinyImageNetDataset(Dataset):
    """ Scenario Dataset for  it requires a scenario number  """
    
    def __init__(self, data_path, transform=None,train=True):
        
        self.train = train
        self.data_path = data_path+'/tiny-imagenet-200/'

        self.transform = transform

        self.classes, self.class_to_idx = find_classes(self.data_path+'train/')
        self._set_data_and_labels()

        

    def _set_data_and_labels(self):
        """ Retrieve all paths and labels and shuffle them"""

        # Retrieve all paths of the specified shenario
        self.paths_train = glob.glob(self.data_path+'train/*/images/*.JPEG')
        self.labels_train = self._extract_labels_from_paths(self.paths_train)

        self.paths_test = glob.glob(self.data_path+'val/images/*/*.JPEG')
        self.labels_test = self._extract_labels_from_paths(self.paths_test)
     
    
    def reset_object_to(self, object_n):
        """ Reset the dataset to a new scenario"""
        self.object_n = object_n+1
        self._set_data_and_labels()

    def _extract_labels_from_paths(self, paths):
        labels = []
        for path in paths:
            if 'train' in path:
                class_current = path.split('_')[0].split('/')[-1]
            else:
                class_current = path.split('/')[-2]
            
            labels.append(self.class_to_idx[class_current])
        return labels
    
    def __len__(self):
        if self.train:
            return len(self.paths_train)
        else:
            return len(self.paths_test)

    def __getitem__(self, index):
        if self.train:
            x = cv2.imread(self.paths_train[index])
            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)

            y = self.labels_train[index]
        else:
            x = cv2.imread(self.paths_test[index])
            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
            y = self.labels_test[index]

        if self.transform:
            x = self.transform(x)

        return [x, y]


def get_all_tinyImageNet_data(path, n_tasks):
    """ Retrieve all paths and labels and shuffle them"""
    #path to dataset 
    dset = tinyImageNetDataset(path, train= True)
    dset_test = tinyImageNetDataset(path, train= False)

    tmp = np.arange(200)
    np.random.shuffle(tmp) #shuffle if you want tasks with non ordered classes

    # split tmp in n_tasks
    tasks = np.split(tmp, n_tasks)

    x_train = []
    y_train = []
    t_train = []
    
    x_test = []
    y_test = []
    t_test = []

    #collect form training
    images = []
    labels= []
    for index in range(dset.__len__()):
        out = dset.__getitem__(index)
        images.append(out[0])
        labels.append(out[1]) 

    x = np.stack(images, axis=0)
    y = np.stack(labels, axis=0)
    
    #collect from val
    images = []
    labels= []
    for index in range(dset_test.__len__()):
        out = dset_test.__getitem__(index)
        images.append(out[0])
        labels.append(out[1]) 

    x_t = np.stack(images, axis=0)
    y_t = np.stack(labels, axis=0)
    
    ###### train 

    for tid, obj_group in enumerate(tasks):
        task_data = []
        task_lbl = []
        task_tlbl = []
        
        cnt=0
        for oid in obj_group:
            
            index = np.where(y==oid)
            # stack all images in a single tensor
            y_buff = np.zeros_like(y[index])+cnt+(tid*200/n_tasks)
            x_buff = x[index]
            t = np.ones_like(y_buff)*tid

            task_data.append(x_buff)
            task_lbl.append(y_buff)
            task_tlbl.append(t)
            cnt+=1

        x_train.append(np.concatenate(task_data, axis=0))
        y_train.append(np.concatenate(task_lbl, axis=0))
        t_train.append(np.concatenate(task_tlbl, axis=0))

    x_train = np.concatenate(x_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    t_train = np.concatenate(t_train, axis=0)

    ###### test

    for tid, obj_group in enumerate(tasks):
        task_data = []
        task_lbl = []
        task_tlbl = []
        cnt=0
        for oid in obj_group:
            
            index = np.where(y_t==oid)
            # stack all images in a single tensor
            y_buff = np.zeros_like(y[index])+cnt+(tid*200/n_tasks)
            x_buff = x_t[index]
            t = np.ones_like(y_buff)*tid

            task_data.append(x_buff)
            task_lbl.append(y_buff)
            task_tlbl.append(t)
            cnt+=1
        x_test.append(np.concatenate(task_data, axis=0))
        y_test.append(np.concatenate(task_lbl, axis=0))
        t_test.append(np.concatenate(task_tlbl, axis=0))

    x_test = np.concatenate(x_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)
    t_test = np.concatenate(t_test, axis=0)

    return (x_train, y_train, t_train), (x_test, y_test, t_test)


if __name__ == '__main__':
    get_all_tinyImageNet_data('/home/leonardolabs/data',n_tasks=10)