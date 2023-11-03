import os
import json
import argparse
import torch
import random
import torch.optim.lr_scheduler
from models import gresnet32, gresnet18, gresnet18mlp
from torch.utils.data import DataLoader
from mind import MIND
from copy import deepcopy
from utils.generic import freeze_model, set_seed, setup_logger
from utils.publisher import push_results
from utils.transforms import to_tensor_and_normalize, default_transforms,default_transforms_core50,\
    to_tensor_and_normalize_core50,default_transforms_TinyImageNet,to_tensor_and_normalize_TinyImageNet, default_transforms_Synbols,to_tensor_and_normalize_Synbols
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
from continuum import ClassIncremental
from continuum.tasks import split_train_val
from continuum.datasets import CIFAR100,Core50
from test_fn import test
import pickle as pkl
from parse import args
from utils.core50dset import get_all_core50_data, get_all_core50_scenario
from utils.tiny_imagenet_dset import get_all_tinyImageNet_data
from utils.synbols_dset import get_synbols_data
from continuum.datasets import InMemoryDataset
from continuum.scenarios import ContinualScenario


def main():
    data_path = os.path.expanduser('~/data')

    # set seed
    set_seed(args.seed)

    # print
    if args.load_model_from_run != '':
        print("#"*30 + f"\n{args.run_name +' loads: '+args.load_model_from_run:^30}\n" +"#"*30)
    else:
        print("#"*30 + f"\n{args.run_name:^30}\n" +"#"*30)

    # log files 
    setup_logger()

    # model
    if args.model == 'gresnet32':
        model = gresnet32(dropout_rate = args.dropout)
    elif args.model == 'gresnet18':
        model = gresnet18(num_classes=args.n_classes)
    elif args.model == 'gresnet18mlp':
        model = gresnet18mlp(num_classes=args.n_classes)
    else:
        raise ValueError("Model not found.")

    if args.load_model_from_run:
        model.load_state_dict(torch.load(f"logs/{args.load_model_from_run}/checkpoints/weights.pt"))
        # load bn weights as pkles 
        bn_weights = pkl.load(open(f"logs/{args.load_model_from_run}/checkpoints/bn_weights.pkl", "rb"))    
        model.bn_weights = bn_weights


    model.to(args.device)

    strategy = MIND(model)

    if args.dataset == 'CIFAR100':
        class_order = list(range(100))
        random.shuffle(class_order)

        train_dataset = CIFAR100(data_path, download=True, train=True)
        test_dataset = CIFAR100(data_path, download=True, train=False)

        strategy.train_scenario = ClassIncremental(
            train_dataset,
            increment=args.classes_per_exp,
            class_order=class_order,
            transformations=default_transforms)

        strategy.test_scenario = ClassIncremental(
            test_dataset,
            increment=args.classes_per_exp,
            class_order=class_order,
            transformations=to_tensor_and_normalize)

    elif 'CORE50' in args.dataset :
        
        if args.dataset == 'CORE50_CI':
            train_data, test_data = get_all_core50_data(data_path, args.n_experiences, split=0.8)
        else:
            train_data, test_data = get_all_core50_scenario(data_path, split=0.8)
        
        train_dataset = InMemoryDataset(*train_data)
        test_dataset = InMemoryDataset(*test_data)

        strategy.train_scenario = ContinualScenario(
            train_dataset,
            transformations=default_transforms_core50)

        strategy.test_scenario = ContinualScenario(
            test_dataset,
            transformations=to_tensor_and_normalize_core50)

    elif args.dataset == 'TinyImageNet':

        train_data, test_data = get_all_tinyImageNet_data(data_path,args.n_experiences)
        train_dataset = InMemoryDataset(*train_data)
        test_dataset = InMemoryDataset(*test_data)

        strategy.train_scenario = ClassIncremental(
            train_dataset,
            increment=args.n_classes//args.n_experiences,
            transformations=default_transforms_TinyImageNet)

        strategy.test_scenario = ClassIncremental(
            test_dataset,
            increment=args.n_classes//args.n_experiences,
            transformations=to_tensor_and_normalize_TinyImageNet)

    elif args.dataset == 'Synbols':

        train_data, test_data = get_synbols_data(data_path, n_tasks=args.n_experiences)
        train_dataset = InMemoryDataset(*train_data)
        test_dataset = InMemoryDataset(*test_data)

        strategy.train_scenario = ClassIncremental(
            train_dataset,
            increment=args.n_classes//args.n_experiences,
            transformations=default_transforms_Synbols)

        strategy.test_scenario = ClassIncremental(
            test_dataset,
            increment=args.n_classes//args.n_experiences,
            transformations=to_tensor_and_normalize_Synbols)



    print(f"Number of classes: {strategy.train_scenario.nb_classes}.")
    print(f"Number of tasks: {strategy.train_scenario.nb_tasks}.")

    if args.load_model_from_run:
        strategy.pruner.masks = torch.load(f"logs/{args.load_model_from_run}/checkpoints/masks.pt")

    for i, train_taskset in enumerate(strategy.train_scenario):
        if args.packnet_original:
            with torch.no_grad():
                strategy.pruner.dezero(strategy.model)

        strategy.experience_idx = i
        strategy.model.set_output_mask(i, train_taskset.get_classes())
        if args.load_model_from_run:
            model.load_bn_params(strategy.experience_idx)

        # prepare dataset
        strategy.train_taskset, strategy.val_taskset = split_train_val(train_taskset, val_split=args.val_split)
        strategy.train_dataloader = DataLoader(strategy.train_taskset, batch_size=args.bsize, shuffle=True)
        if len(strategy.val_taskset):
            strategy.val_dataloader = DataLoader(strategy.val_taskset, batch_size=args.bsize, shuffle=True)
        else:
            strategy.val_dataloader = DataLoader(strategy.test_scenario[i], batch_size=args.bsize, shuffle=True)

        #################### TRAIN ###########################

        # instantiate new model
        if not args.self_distillation:
            if args.model == 'gresnet32':
                strategy.fresh_model = gresnet32(dropout_rate = args.dropout)
            elif args.model == 'gresnet18':
                strategy.fresh_model = gresnet18(num_classes=args.n_classes)
            elif args.model == 'gresnet18mlp':
                strategy.fresh_model = gresnet18mlp(num_classes=args.n_classes)
            else:
                raise ValueError("Model not found.")
        else:
            strategy.fresh_model = deepcopy(strategy.model)
            strategy.distillation = False
            strategy.pruner.set_gating_masks(strategy.fresh_model, strategy.experience_idx, weight_sharing=args.weight_sharing, distillation=strategy.distillation)
        
        strategy.fresh_model.to(args.device)
        strategy.fresh_model.set_output_mask(i, train_taskset.get_classes())

        # instantiate oprimizer
        strategy.train_epochs = args.epochs
        strategy.distillation = False
        strategy.optimizer = torch.optim.AdamW(strategy.fresh_model.parameters(), lr=args.lr, weight_decay=args.wd)
        strategy.scheduler = torch.optim.lr_scheduler.MultiStepLR(strategy.optimizer, milestones=args.scheduler, gamma=0.5, last_epoch=-1, verbose=False)

        print(f'-.-.-.-.-.-. Start training on experience {i+1} - epochs: {strategy.train_epochs} .-.-.-.-.-.')
        strategy.train()

        # Freeze the model for distillation purposes
        strategy.distill_model = freeze_model(deepcopy(strategy.fresh_model))
        strategy.distill_model.to(args.device)

        ########### FINETUNING/DISTILLATION ################
        # selects subset of neurons, prune non selected weights
        if strategy.warmup:
            strategy.optimizer = torch.optim.AdamW(strategy.model.parameters(), lr=args.lr_distillation, weight_decay=args.wd_distillation)
            strategy.warmup()

        if not args.load_model_from_run:
            with torch.no_grad():
                strategy.pruner.prune(strategy.model, strategy.experience_idx, strategy.distill_model, args.self_distillation)


        strategy.train_epochs = args.epochs_distillation
        strategy.distillation = True
        strategy.optimizer = torch.optim.AdamW(strategy.model.parameters(), lr=args.lr_distillation, weight_decay=args.wd_distillation)
        strategy.scheduler = torch.optim.lr_scheduler.MultiStepLR(strategy.optimizer, milestones=args.scheduler_distillation, gamma=0.5, last_epoch=-1, verbose=False)
        print(f"    >>> Start Finetuning epochs: {args.epochs_distillation} <<<")
        strategy.pruner.set_gating_masks(strategy.model, strategy.experience_idx, weight_sharing=args.weight_sharing, distillation=strategy.distillation)
        strategy.train()

        #################### TEST ##########################
        # concatenate pytorch datasets up to the current experience
        with torch.no_grad():
            # write accuracy on the test set
            total_acc = 0
            task_acc = 0
            accuracy_e = 0
            total_acc, task_acc, accuracy_e, accuracy_taw = test(strategy, strategy.test_scenario[:i+1])

            with open(f"logs/{args.run_name}/results/total_acc.csv", "a") as f:
                f.write(f"{strategy.experience_idx},{total_acc:.4f}\n")
            with open(f"logs/{args.run_name}/results/total_acc_taw.csv", "a") as f:
                f.write(f"{strategy.experience_idx},{accuracy_taw:.4f}\n")

        # save the model and the masks
        if not args.load_model_from_run:
            torch.save(strategy.model.state_dict(), f"logs/{args.run_name}/checkpoints/weights.pt")
            torch.save(strategy.pruner.masks, f"logs/{args.run_name}/checkpoints/masks.pt")
            pkl.dump(strategy.model.bn_weights, open(f"logs/{args.run_name}/checkpoints/bn_weights.pkl", "wb"))

    # push results to excel
    push_results(args, total_acc, task_acc, accuracy_e, accuracy_taw)


if __name__ == "__main__":

    main()

