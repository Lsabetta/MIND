#!/bin/bash

python main.py --run_name "test" \
        --dataset "CIFAR100" \
        --cuda 0 \
        --seed 0 \
        --n_experiences 10 \
        --model "gresnet32" \
        --epochs 50 \
        --temperature 6.5 \
        --epochs_distillation 50 \
        --num_workers 4 \

