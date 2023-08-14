#!/bin/bash

# Experiments Table 1 (A) - Synbols
for seed in 0 1 2 3 4 5 6 7 8 9;
do
    python main.py --run_name "synbols_experiment" \
            --dataset "Synbols" \
            --cuda 0 \
            --seed $seed \
            --n_experiences 10 \
            --model "gresnet32" \
            --epochs 25 \
            --lr 0.005 \
            --scheduler 10 20 \
            --epochs_distillation 25 \
            --lr_distillation 0.035 \
            --scheduler_distillation 15 \
            --temperature 4
done