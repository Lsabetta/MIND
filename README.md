# MIND
Authors: Jacopo Bonato, Francesco Pelosin, Luigi Sabetta, Alessandro Nicolosi

Preprint: 
https://arxiv.org/abs/2312.02916

# Installation

- Step 1:

    Build the docker image from the Dockerfile : `docker build -f Dockerfile -t mind:1.0 . `

- Step 2:

    Run your image : `docker run -it --gpus all -v "/path_to_dataset_folder":/root/data -v "/path_to_MIND_folder":/MIND mind:1.0 /bin/bash`


# Experiements

For each dataset we make available a bashscript (i.e. Cifar100_exp.sh) containing the parameters used to obtain the results reported in the paper. Each script will run 10 experiemnts with 10 different seeds.

### Data Path setup
Before running everything, go to main.py and set the data path (line number 31) to the folder where you want to store the data. The data will be either downloaded automatically from continuum or needs to be downloaded there.


### CIFAR100/10 (works out of the box)
To run the experiments on CIFAR100/10 dataset in class incremental (Table 1.A), run the following command:

```
sh cifar100_exp.sh
```
all the outputs will be logged in the `./logs/cifar100_experiment` folder (accuracies, losses, plots) and in the terminal.

---
### CORE50_CI/10 (requires setup)
Download the dataset and unzip it into your dataset folder (the folder that you have mounted into your docker image 'path_to_dataset_folder')
```
wget download the dataset from http://bias.csr.unibo.it/maltoni/download/core50/core50_128x128.zip
unzip core50_128x128.zip 
```

- Step 1:

    Before running the experiments on core50 dataset, you need to resize the dataset and prepare it for continuum with the following command. _NOTE: please read the script before running to understand how to set the paths:_
    ```
    cd ./utils
    python resize_core50.py
    ```

- Step 2:

    To run the experiments on CORE50/10 dataset in class incremental (Table 1.A), run:
    ```
    sh core50ci_exp.sh
    ```
    all the outputs will be logged in the `./logs/core_experiment` folder (accuracies, losses, plots) and in the terminal.

---
### TinyImgNet/10 (requires setup)

Download the dataset and unzip it intop your dataset folder.
```
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip
```
 Then we need to prepare the folder structure to work with continuum as follows:


- Step 1:

    ```
    cd ./utils
    python tiny_imgnet_setup.py --data_dir /path/to/data --dataset tiny-imagenet-200
    ```
    _NOTE: please read the script before running to understand how it works_

- Step 2:
    To run the experiments (Table 1.A), run:

    ```
    sh tinyimgnet_exp.sh
    ```
    all the outputs will be logged in the `./logs/tinyimgnet_experiment` folder (accuracies, losses, plots) and in the terminal.

---
### Synbols (requires generation)

- Setup 1 (outside our docker): 

    In order to run the experiments on Synbols dataset, we need to generate the dataset first. To do so, we need to install the synbols package from the [official repository](https://github.com/ServiceNow/synbols).

- Setup 2 (outside our docker):

    Then we can generate the dataset by running the notebook `./utils/generate_synbols.ipynb`. The script will output two zip that you will put in your data folder. That's enough, the experiments will take care of everything.

- Step 3 (inside our docker):
    
    To run the experiments (Table 1.A), run:

    ```
    sh synbols_exp.sh
    ```
    all the outputs will be logged in the `./logs/synbols_experiment` folder (accuracies, losses, plots) and in the terminal.