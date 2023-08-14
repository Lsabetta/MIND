# MIND

## Installation
Build the docker image from the Dockerfile

docker build -f Dockerfile -t mind:1.0 . 

Run your image 

docker run -it --gpus all -v "/path_to_MIND_folder":/MIND mind:1.0 /bin/bash

## Experiemnts
For each dataset we make available a bashscript (i.e. Cifar100_exp.sh) containing the parameters used to obtain the results reported in the paper. Each script will run 10 experiemnts with 10 different seeds.


