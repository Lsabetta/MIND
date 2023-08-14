#!/bin/bash

# Experiments Table 1 (A) - TinyImageNet
for seed in 0 1 2 3 4 5 6 7 8 9;
do
    python main.py --run_name "tinyimgnet_experiment" \
            --dataset "TinyImageNet" \
            --cuda 0 \
            --seed $seed \
            --n_experiences 10 \
            --model "gresnet32" \
            --epochs 100 \
            --lr 0.005 \
            --scheduler 70 90 \
            --epochs_distillation 120 \
            --lr_distillation 0.035 \
            --scheduler_distillation 80 110 \
            --temperature 12
done