
import cv2
import random
import glob
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Subset, Dataset, ConcatDataset

###############################################################################
######################## DOMAIN INCREMENTAL ###################################
###############################################################################
class Core50Dataset(Dataset):
    """ Scenario Dataset for Core50 it requires a scenario number  """
    
    def __init__(self, data_path, object_n, transform=None):
        self.data_path = data_path+'/core50_64x64/'
        self.transform = transform
        self.object_n = object_n+1
        self._set_data_and_labels()

    def _set_data_and_labels(self):
        """ Retrieve all paths and labels and shuffle them"""

        # Retrieve all paths of the specified shenario
        self.paths = glob.glob(self.data_path+'/*/'+f'o{self.object_n}/*.png')
        self.labels = self._extract_labels_from_paths(self.paths)
        
        # Shuffle the lists in unison
        combined = list(zip(self.paths, self.labels))
        random.shuffle(combined)
        self.paths, self.labels = zip(*combined)

        # Retrieve all
        #self.paths[-1] = glob.glob(self.data_path+'/*/*/*.png')        
    
    def reset_object_to(self, object_n):
        """ Reset the dataset to a new scenario"""
        self.object_n = object_n+1
        self._set_data_and_labels()

    def _extract_labels_from_paths(self, paths):
        labels = []
        for path in paths:
            # Corrects labels starting from 0 to 49
            labels.append(int(path.split('/')[-2][1:])-1)
        return labels
    
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):

        x = cv2.imread(self.paths[index])
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)

        y = self.labels[index]
        if self.transform:
            x = self.transform(x)

        return x, y


def get_all_core50_data(path, n_tasks, split):
    """ Retrieve all paths and labels and shuffle them"""
    dset = Core50Dataset(path, 0)

    tmp = np.arange(50)
    np.random.shuffle(tmp)

    # split tmp in n_tasks
    tasks = np.split(tmp, n_tasks)

    x_all = []
    y_all = []
    t_all = []
    for tid, obj_group in enumerate(tasks):

        # remap labels obj_group to tid*10

        task_data = []
        task_lbl = []
        task_tlbl = []
        for i, oid in enumerate(obj_group):
            dset.reset_object_to(oid)

            # stack all images in a single tensor
            x = np.stack([x for x, y in dset], axis=0)
            y = np.stack([tid*len(obj_group)+i for x, y in dset], axis=0)
            t = np.ones_like(y)*tid

            task_data.append(x)
            task_lbl.append(y)
            task_tlbl.append(t)

        x_all.append(np.concatenate(task_data, axis=0))
        y_all.append(np.concatenate(task_lbl, axis=0))
        t_all.append(np.concatenate(task_tlbl, axis=0))

    x_all = np.concatenate(x_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    t_all = np.concatenate(t_all, axis=0)
    



    # shuffle all data
    idx = np.arange(len(x_all))
    np.random.shuffle(idx)

    x_all = x_all[idx]
    y_all = y_all[idx]
    t_all = t_all[idx]

    # split in train and test
    n_train = int(len(x_all)*split)
    x_train = x_all[:n_train]
    y_train = y_all[:n_train]
    t_train = t_all[:n_train]

    x_test = x_all[n_train:]
    y_test = y_all[n_train:]
    t_test = t_all[n_train:]

    return (x_train, y_train, t_train), (x_test, y_test, t_test)


def remap(y_train, y_test, t_train, t_test):
    # remap labels for each first task to [0, 9] then seconnd task to [10, 19] etc.

    for tid in np.unique(t_train):

        # get all labels for this task
        idx = t_train == tid
        y = y_train[idx]

        # remap labels
        



###############################################################################
######################## DOMAIN INCREMENTAL ###################################
###############################################################################


class Core50DatasetScenario(Dataset):
    """ Scenario Dataset for Core50 it requires a scenario number  """
    
    def __init__(self, data_path, transform=None):
        self.data_path = data_path+'/core50_64x64/'
        self.transform = transform
        self.scenario_n = -1

    def _set_data_and_labels(self):
        """ Retrieve all paths and labels and shuffle them"""

        # Retrieve all paths of the specified shenario
        self.paths = glob.glob(self.data_path+'/'+f's{self.scenario_n}/*/*.png')
        self.labels = self._extract_labels_from_paths(self.paths)
        
        # Shuffle the lists in unison
        combined = list(zip(self.paths, self.labels))
        assert len(combined) > 0, 'brez'
        random.shuffle(combined)
        self.paths, self.labels = zip(*combined)
        assert len(self.paths) == len(self.labels) and len(self.paths) > 0, 'brez'

        # Retrieve all
        #self.paths[-1] = glob.glob(self.data_path+'/*/*/*.png')        
    
    def reset_scenario_to(self, scenario_n):
        """ Reset the dataset to a new scenario"""
        self.scenario_n = scenario_n
        self._set_data_and_labels()

    def _extract_labels_from_paths(self, paths):
        labels = []
        for path in paths:
            # Corrects labels starting from 0 to 49
            labels.append(int(path.split('/')[-2][1:])-1)
        return labels
    
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):

        x = cv2.imread(self.paths[index])
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)

        y = self.labels[index]
        if self.transform:
            x = self.transform(x)

        return x, y


def get_all_core50_scenario(path, split):
    """ Retrieve all paths and labels and shuffle them"""

    s_data = Core50DatasetScenario(path)
    scenario_id = [x for x in range(0,11)]
    random.shuffle(scenario_id)

    x_all = []
    y_all = []
    t_all = []
    for sid in scenario_id:
        
        s_data.reset_scenario_to(sid+1)
        
        # stack all images in a single tensor
        x = np.stack([x for x, y in s_data], axis=0)
        y = np.stack([y for x, y in s_data], axis=0)
        t = np.ones_like(y)*sid

        x_all.append(x)
        y_all.append(y)
        t_all.append(t)
    
    # stack all scenarios
    x_all = np.concatenate(x_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    t_all = np.concatenate(t_all, axis=0)

    # shuffle all data
    idx = np.arange(len(x_all))
    np.random.shuffle(idx)

    x_all = x_all[idx]
    y_all = y_all[idx]
    t_all = t_all[idx]

    # split in train and test
    n_train = int(len(x_all)*split)
    x_train = x_all[:n_train]
    y_train = y_all[:n_train]
    t_train = t_all[:n_train]

    x_test = x_all[n_train:]
    y_test = y_all[n_train:]
    t_test = t_all[n_train:]

    return (x_train, y_train, t_train), (x_test, y_test, t_test)


if __name__ == '__main__':
    import os
    data_path = os.path.expanduser('~/data')
    out = get_all_core50_data(data_path, 10, 0.8)