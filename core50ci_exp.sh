#!/bin/bash

# Experiments Table 1 (A) - CORE50-CI
for seed in 0 1 2 3 4 5 6 7 8 9;
do
    python main.py --run_name "core50_experiment" \
            --dataset "CORE50_CI" \
            --cuda 0 \
            --seed $seed \
            --n_experiences 10 \
            --model "gresnet32" \
            --epochs 20 \
            --lr 0.005 \
            --scheduler 15 \
            --epochs_distillation 20 \
            --lr_distillation 0.035 \
            --scheduler_distillation 15 \
            --temperature 3
done