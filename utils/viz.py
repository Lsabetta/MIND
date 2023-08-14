import os
import math
import matplotlib.pyplot as plt
import seaborn as sns


def check_paths(run_name):
    os.makedirs(f'logs/{run_name}/confmats/distillation_phase_True', exist_ok=True)
    os.makedirs(f'logs/{run_name}/confmats/distillation_phase_False', exist_ok=True)

    os.makedirs(f'logs/{run_name}/gradients/distillation_phase_True', exist_ok=True)
    os.makedirs(f'logs/{run_name}/gradients/distillation_phase_False', exist_ok=True)


def plt_test_confmat(run_name, confusion_mat, experience_idx):
    check_paths(run_name)
    
    sns.heatmap(confusion_mat, annot=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')

    accuracy = confusion_mat.diag().sum() / confusion_mat.sum()
    plt.title(f'Test accuracy: {accuracy}')

    #plt.show()
    plt.savefig(f'logs/{run_name}/confmats/test_only_task_{experience_idx+1:03}.png')
    plt.close()


def plt_test_confmat_task(run_name, confusion_mat):
    check_paths(run_name)
    
    sns.heatmap(confusion_mat, annot=True)
    plt.xlabel('Test')
    plt.ylabel('Train')

    plt.title("Task Confusion Matrix")
    #plt.show()
    plt.savefig(f'logs/{run_name}/confmats/task_confmat.png')
    plt.close()  
    

def plt_confmats(run_name, train_conf_mat, test_conf_mat, distillation, experience_idx):
    check_paths(run_name)
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    sns.heatmap(train_conf_mat, annot=False, ax=axs[0])
    sns.heatmap(test_conf_mat, annot=False, ax=axs[1])
    
    axs[0].set_title('Train')
    axs[1].set_title('Test')
    axs[0].set_xlabel('Predicted')
    axs[1].set_xlabel('Predicted')
    axs[0].set_ylabel('True')
    axs[1].set_ylabel('True')

    train_acc = train_conf_mat.diag().sum() / train_conf_mat.sum()
    test_acc = test_conf_mat.diag().sum() / test_conf_mat.sum()

    plt.suptitle(f"tr_acc:{train_acc:.4f}, te_acc:{test_acc:.4f}")
    plt.tight_layout()
    plt.savefig(f'logs/{run_name}/confmats/distillation_phase_{distillation}/{experience_idx+1:03}.png')
    #plt.show()
    plt.close()


def plt_masks_grad_weight(run_name, model, pruner, experience_idx, distillation, layer_idx=1):
    check_paths(run_name)

    layer_mask = pruner.masks[layer_idx]
    for i, module in enumerate(model.modules()):
        if i == layer_idx:
            break
    dim1 = module.weight.shape[0]
    dim2 = math.prod(list(module.weight.shape[1:]))

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15,5))
    sns.heatmap(layer_mask.view(dim1, dim2).cpu().numpy(), annot=True, vmin=-1, vmax=9, cmap=sns.color_palette("muted", 10), ax=axs[0])

    grads = module.weight.grad.view(dim1, dim2).cpu().detach().numpy()
    sns.heatmap(grads, annot=False, vmin=grads.min(), vmax=grads.max(), ax=axs[1])

    weights = module.weight.view(dim1, dim2).cpu().detach().numpy()
    sns.heatmap(weights, annot=False, vmin=weights.min(), vmax=weights.max(), ax=axs[2])
    
    # set titles
    axs[0].set_title("Mask")
    axs[1].set_title("Gradients")
    axs[2].set_title("Weights")

    #module_path = module.__str__().split("(")[0]
    module_path = f"logs/{run_name}/gradients/distillation_phase_{distillation}/mod{layer_idx:04}"
    if not os.path.exists(module_path):
        os.makedirs(module_path)

    plt.savefig(f"{module_path}/{experience_idx+1:03}.png")

    plt.close()

