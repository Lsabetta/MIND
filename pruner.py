import torch
import torch.nn as nn
from models.gated_resnet32 import GatedConv2d, GatedLinear
from parse import args
import math

class Pruner(object):
    """Performs pruning on the given model."""

    def __init__(self, model, train_bias, train_bn, warmup):
        self.train_bias = train_bias
        self.train_bn = train_bn

        # -1 means that the weight should accept gradients
        self.masks = {}
        for module_idx, module in enumerate(model.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                mask = torch.ones(module.weight.size(), dtype=torch.int8)
                self.masks[module_idx] = mask * -1
        self.warmup = warmup

    def select_random_weights(self, weights, mask):

        # get the number of weights to prune linearly
        n_random = weights.numel() // args.n_experiences

        active_weights = (mask == -1)

        if n_random > active_weights.sum():
            n_random = active_weights.sum()

        # get the indices of the active weights
        active_indices = torch.nonzero(active_weights)

        # permute the indices and select n_random
        random_indices = active_indices[torch.randperm(active_indices.size(0))][:n_random]

        # return mask which has true in random indices
        mask_out = torch.zeros_like(mask).to(torch.bool)
        # put 1 in the indices of the random weights
        mask_out[[random_indices[:, x] for x in range(random_indices.shape[1])]] = True

        return mask_out


    def ripristinate_weights(self, model, old_model, experience_idx, distillation):
        """Ripristinates the gradients of the new model from the old model."""

        for module_idx, (module, old_module) in enumerate(zip(model.modules(), old_model.modules())):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                layer_mask = self.masks[module_idx]
                # if distillation:
                if args.self_distillation and not distillation:
                    module.weight[layer_mask != -1] = old_module.weight[layer_mask != -1]
                else:
                    module.weight[layer_mask != experience_idx] = old_module.weight[layer_mask != experience_idx]

    def most_important_weights_mask(self, weights, mask):

        # get the number of weights to prune linearly
        n_important = weights.numel() // args.n_experiences

        active_weights = (mask == -1)

        if n_important > active_weights.sum():
            n_important = active_weights.sum()

        cutoff_value = weights[active_weights].abs().sort()[0][-n_important]
        
        if cutoff_value == 0.0:
            weights[ (weights == 0.0) & active_weights ] += torch.rand(weights[ (weights == 0.0) & active_weights ].shape).to(weights.device)*1e-5
            cutoff_value = weights[active_weights].abs().sort()[0][-n_important]

        # Remove those weights which are below cutoff and have been trained now
        important = ((weights.abs()) >= cutoff_value) * active_weights
        return important


    def prune(self, model, experience_idx, distill_model, self_distillation):
        n_modules = len([m for m in model.modules()])
        for module_idx, (module, distill_module) in enumerate(zip(model.modules(), distill_model.modules())):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                self.masks[module_idx] = self.masks[module_idx].to(module.weight.device)
                #if it is the last layer, assign to the mask only the weights connecting to the new classes
                if module_idx == n_modules-2 and args.dataset != 'CORE50' and not args.self_distillation:
                    if not self.warmup:
                        subset = torch.zeros_like(self.masks[module_idx])
                        subset[experience_idx*args.classes_per_exp:(experience_idx+1)*args.classes_per_exp, :] = 1
                        #convert subset to bool
                        subset = subset.to(torch.bool)
                    else:
                        subset = self.most_important_weights_mask(module.weight, self.masks[module_idx])
                else:
                    if args.self_distillation:
                        subset= self.most_important_weights_mask(module.weight, self.masks[module_idx])
                    else:
                        if not self.warmup:
                            subset= self.select_random_weights(module.weight, self.masks[module_idx])
                        else:
                            subset = self.most_important_weights_mask(module.weight, self.masks[module_idx])
                self.masks[module_idx][subset] = experience_idx

                #when self distillation, set fresh model weights as starting point
                if self_distillation:
                    module.weight[self.masks[module_idx] == experience_idx] = distill_module.weight[self.masks[module_idx] == experience_idx]
                

                # Set unassigned weights to 0   
                module.weight[self.masks[module_idx] == -1] = 0.0
                if not self_distillation:
                    if not args.weight_sharing:
                        if isinstance(module, nn.Conv2d):
                            n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                            module.weight[self.masks[module_idx] == experience_idx] = torch.randn(module.weight[self.masks[module_idx] == experience_idx].shape).to(module.weight.device)*math.sqrt(2./n)
                        if isinstance(module, nn.Linear):
                            n = module.weight.shape[0]*module.weight.shape[1]
                            module.weight[self.masks[module_idx] ==  experience_idx] = torch.randn(module.weight[self.masks[module_idx] == experience_idx].shape).to(module.weight.device)*math.sqrt(2./n)

    def dezero(self, model):
        for module_idx, module in enumerate(model.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                self.masks[module_idx] = self.masks[module_idx].to(module.weight.device)
                # Set unassigned weights to 0
                if isinstance(module, nn.Conv2d):
                    n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                    module.weight[self.masks[module_idx] == -1] = torch.randn(module.weight[self.masks[module_idx] == -1].shape).to(module.weight.device)*math.sqrt(2./n)
                if isinstance(module, nn.Linear):
                    n = module.weight.shape[0]*module.weight.shape[1]
                    module.weight[self.masks[module_idx] == -1] = torch.randn(module.weight[self.masks[module_idx] == -1].shape).to(module.weight.device)*math.sqrt(2./n)


    def set_gating_masks(self, model, task_id, weight_sharing, distillation):
        # set the mask for each layer
        for module_idx, module in enumerate(model.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                layer_mask = self.masks[module_idx]
                if distillation:
                    if weight_sharing:
                        module.mask = torch.logical_and(layer_mask <= task_id, layer_mask != -1).to(module.weight.device)
                    else:
                        module.mask = (layer_mask == task_id).to(torch.float32).to(module.weight.device) 

                else:
                    module.mask = torch.ones_like(module.mask).to(torch.float32).to(module.weight.device)

