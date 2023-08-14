import random
import numpy as np
import torch
import argparse
import json
import os
from parse import args

def setup_logger():
    """ creates files for logging in logs/ with seed"""

    # create run folder
    os.makedirs(f"logs/{args.run_name}", exist_ok=True)
    os.makedirs(f"logs/{args.run_name}/results", exist_ok=True)
    os.makedirs(f"logs/{args.run_name}/checkpoints", exist_ok=True)
    os.makedirs(f"logs/{args.run_name}/confmats", exist_ok=True)
    os.makedirs(f"logs/{args.run_name}/gradients", exist_ok=True)

    # headers
    with open(f"logs/{args.run_name}/results/acc.csv", "w") as f:
        f.write(f"experience_idx,distillation,epoch,acc_train,acc_test\n")

    with open(f"logs/{args.run_name}/results/loss.csv", "w") as f:
        f.write(f"experience_idx,distillation,epoch,ce,distill\n")

    with open(f"logs/{args.run_name}/results/total_acc.csv", "w") as f:
        f.write(f"experience_idx,acc\n")

    with open(f"logs/{args.run_name}/results/total_acc_taw.csv", "w") as f:
        f.write(f"experience_idx,acc_taw\n")

    # dumps parameters
    tmp = args.device
    args.device = None
    with open(f'logs/{args.run_name}/params.json', 'w') as f:
        json.dump(vars(args), f)
    args.device = tmp


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resotre_arguments_from_json(json_file):
    with open(json_file, 'r') as f:
        args_dict = json.load(f)

    parser = argparse.ArgumentParser()

    for arg_name, arg_value in args_dict.items():
        parser.add_argument('--' + arg_name, type=type(arg_value), default=arg_value)

    args = parser.parse_args()

    return args


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return model


def unfreeze_model(model):
    for param in model.parameters():
        param.requires_grad = True
    return model
