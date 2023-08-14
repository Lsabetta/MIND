#!/bin/bash

# Experiments Table 1 (A) - cifar100

for seed in 0 1 2 3 4 5 6 7 8 9;
do
    python main.py --run_name "cifar100_experiment" \
            --dataset "CIFAR100" \
            --cuda 0 \
            --seed $seed \
            --n_experiences 10 \
            --model "gresnet32" \
            --epochs 50 \
            --lr 0.005 \
            --scheduler 35 \
            --epochs_distillation 50 \
            --lr_distillation 0.035 \
            --scheduler_distillation 40 \
            --temperature 6.5 
done