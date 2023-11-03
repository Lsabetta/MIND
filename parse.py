import argparse
import torch

dset_stats = {
    'CIFAR100': {'n_classes': 100},
    'CIFAR10': {'n_classes': 10},
    'CORE50': {'n_classes': 50},
    'CORE50_CI': {'n_classes': 50},
    'TinyImageNet':{'n_classes':200},
    'Synbols':{'n_classes':200}
    }


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="test")
    parser.add_argument("--dataset", type=str, default="CIFAR100")
    parser.add_argument("--cuda", type=int, default=0, help="Select zero-indexed cuda device. -1 to use CPU.")
    parser.add_argument("--temperature", type=float, default=6.5)

    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_split", type=float, default=0.0)
    parser.add_argument("--bsize", type=int, default=256)
    parser.add_argument("--n_experiences", type=int, default=10)
    parser.add_argument("--weight_sharing", action ="store_false")
    parser.add_argument("--model", type=str, default='gresnet32')
    parser.add_argument("--load_model_from_run", type=str, default='')

    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--epochs", type=int, default=4) # <------- epochs train
    parser.add_argument("--scheduler", type=int, nargs='+', default=[35,50])

    parser.add_argument("--wd_distillation", type=float, default=0)
    parser.add_argument("--lr_distillation", type=float, default=0.035)
    parser.add_argument("--epochs_distillation", type=int, default=4) # <------- epochs distillation
    parser.add_argument("--distill_beta", type=float, default=5)
    parser.add_argument("--scheduler_distillation", type=int, nargs='+', default=[40,60])
    parser.add_argument("--details", type=str, nargs ='+', default="baseline cifar100")
    parser.add_argument("--dropout", type=float, default=0.,help="Select prob of dropout")
    parser.add_argument("--self_distillation", action ="store_true")
    parser.add_argument("--packnet_original", action ="store_true")
    parser.add_argument("--no_bn", action ="store_true")
    parser.add_argument("--plot_gradients", action ="store_false")
    parser.add_argument("--log_every", type=int, default=2)
    parser.add_argument("--plot_gradients_of_layer", type=int, default=1)
    parser.add_argument("--distill_loss", type=str, default="JS")

    

    options = parser.parse_args()

    ##### post processing arguments #####

    # append seed to run_name
    options.run_name = options.run_name + f"_{options.seed}"
    if options.load_model_from_run:
        options.load_model_from_run = options.load_model_from_run + f"_{options.seed}"

    options.device = torch.device(f"cuda:{options.cuda}" if torch.cuda.is_available() and options.cuda >= 0 else "cpu")
    options.n_classes = dset_stats[options.dataset]['n_classes']
    if options.packnet_original:
        options.self_distillation = True
        options.distill_beta = 0.
    if options.dataset == 'CORE50' and options.datasset != 'CORE50_CI':
        options.classes_per_exp = options.n_classes
        options.n_experiences = 11
    else:
        options.classes_per_exp = options.n_classes // options.n_experiences
    print(options)
    return options

args = get_args()
 
