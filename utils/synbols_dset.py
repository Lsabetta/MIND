
import numpy as np
import random

def remap(class_indices, tasks):
    """ remap class indices to new indices
    class_indices: np array of class indices
    tasks: list of lists of class indices
    """

    concat_tasks = np.concatenate(tasks)
    remapped_indices = np.zeros(class_indices.shape[0]).astype(np.int16)
    for i in range(class_indices.shape[0]):
        remapped_indices[i] = np.where(concat_tasks == class_indices[i])[0][0]
    return remapped_indices
    

def convert_to_int(y):
    """ convert ideograms into int labels
    y: list of dicts
    """
    labels_data = np.zeros(len(y)).astype(np.int16)
    y_char = [yi['char'] for yi in y]
    all_chars = sorted(list(set(y_char)))
    for i in range(len(y)):
        labels_data[i] = all_chars.index(y[i]['char']) 
    return labels_data


def split_to_tasks(x, y, tasks, n_tasks):
    """ split data into tasks
    x: np array of images
    y: np array of labels
    tasks: list of lists of class indices
    n_tasks: number of tasks
    """
    x_out = []
    y_out = []
    t_out = []

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
            
        x_out.append(np.concatenate(task_data, axis=0))
        y_out.append(np.concatenate(task_lbl, axis=0))
        t_out.append(np.concatenate(task_tlbl, axis=0))

    x_out = np.concatenate(x_out, axis=0)
    y_out = np.concatenate(y_out, axis=0)
    t_out = np.concatenate(t_out, axis=0)

    return x_out, y_out, t_out


def get_synbols_data(data_path, n_tasks):
    """
    n_classes depends on the number of classes in the dataset, we set the default to 200
    """
    n_classes = 200

    # load data
    train = np.load(data_path + '/Synbols/train.npz', allow_pickle=True)
    x_train, y_train = train['x_train'], train['y_train']

    test = np.load(data_path + '/Synbols/test.npz', allow_pickle=True)
    x_test, y_test = test['x_test'], test['y_test']

    # val = np.load(data_path + '/Synbols/val.npz')
    # x_val, y_val = val['x_val'], val['y_val']

    y_train = convert_to_int(y_train)
    y_test = convert_to_int(y_test)

    # create tasks and shuffle
    tasks = [x for x in range(0, n_classes)]
    random.shuffle(tasks)
    tasks = np.array_split(tasks, n_tasks)
    
    # remap
    y_train = remap(y_train, tasks)
    y_test = remap(y_test, tasks)

    # split into tasks
    tr_x, tr_y, tr_t = split_to_tasks(x_train, y_train, tasks, n_tasks)
    te_x, te_y, te_t = split_to_tasks(x_test, y_test, tasks, n_tasks)

    return (tr_x, tr_y, tr_t), (te_x, te_y, te_t)


if __name__ == '__main__':
    seed = 10
    random.seed(seed)
    np.random.seed(seed)
    # debug
    (tr_x, tr_y, tr_t), (te_x, te_y, te_t) = get_synbols_data('/home/pelosinf/data/', 100)

    print('done')
