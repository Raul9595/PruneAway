import os
import shutil
import argparse
import numpy as np
import copy
import torch
import torchvision
import torch_pruning as tp

from cifar import load_cifar, eval_cifar
from models import *
from utils import load_model
from pruning import prune_model_structured


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', type=str, default='cifar', help='Evaluation dataset')
    parser.add_argument('--model', type=str, default='resnet50', help='Evaluation dataset')
    opt = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = load_model(device, opt.model)

    if model is None:
        pass

    else:
        model_sp = copy.deepcopy(model)
        
        if opt.eval == 'cifar':
            # Loading CIFAR data
            cifar_testloader = load_cifar('test')
            cifar_trainloader = load_cifar('train')

            # Original parameters
            org_params = sum([np.prod(p.size()) for p in model.parameters()])

            print('\nEvaluating original model')
            org_param_str = "\nNumber of Original Parameters: %.1fM"%(org_params/1e6)
            print(org_param_str)

            # Evaluate dataset
            acc, avg_inf_time = eval_cifar(model, cifar_testloader, device)
            print(f'\nAccuracy - {acc}')
            print(f'Average Inference Time - {avg_inf_time}')
            print('\n*********************************************\n')

            ##############################################################################

            # Pruning model (Structured)
            model_sp_new = copy.deepcopy(model_sp)
            prune_model_structured(model_sp_new, device, 0.015)

            # Pruned parameters
            pr_params = sum([np.prod(p.size()) for p in model_sp_new.parameters()])

            print('Evaluating pruned model (Structured at 0.015)')
            pr_param_str = "\nNumber of Pruned Parameters: %.1fM"%(pr_params/1e6)
            print(pr_param_str)

            # Evaluate dataset
            acc, avg_inf_time = eval_cifar(model_sp_new, cifar_testloader, device)
            print(f'\nAccuracy - {acc}')
            print(f'Average Inference Time - {avg_inf_time}')
            print('\n*********************************************\n')

            ##############################################################################

            # Pruning model (Structured)
            model_sp_new = copy.deepcopy(model_sp)
            prune_model_structured(model_sp_new, device, 0.02)

            # Pruned parameters
            pr_params = sum([np.prod(p.size()) for p in model_sp_new.parameters()])

            print('Evaluating pruned model (Structured at 0.02)')
            pr_param_str = "\nNumber of Pruned Parameters: %.1fM"%(pr_params/1e6)
            print(pr_param_str)

            # Evaluate dataset
            acc, avg_inf_time = eval_cifar(model_sp_new, cifar_testloader, device)
            print(f'\nAccuracy - {acc}')
            print(f'Average Inference Time - {avg_inf_time}')
            print('\n*********************************************\n')

            ##############################################################################

            # Pruning model (Structured)
            model_sp_new = copy.deepcopy(model_sp)
            prune_model_structured(model_sp_new, device, 0.025)

            # Pruned parameters
            pr_params = sum([np.prod(p.size()) for p in model_sp_new.parameters()])

            print('Evaluating pruned model (Structured at 0.025)')
            pr_param_str = "\nNumber of Pruned Parameters: %.1fM"%(pr_params/1e6)
            print(pr_param_str)

            # Evaluate dataset
            acc, avg_inf_time = eval_cifar(model_sp_new, cifar_testloader, device)
            print(f'\nAccuracy - {acc}')
            print(f'Average Inference Time - {avg_inf_time}')
            print('\n*********************************************\n')

            ##############################################################################

            # Pruning model (Structured)
            model_sp_new = copy.deepcopy(model_sp)
            prune_model_structured(model_sp_new, device, 0.03)

            # Pruned parameters
            pr_params = sum([np.prod(p.size()) for p in model_sp_new.parameters()])

            print('Evaluating pruned model (Structured at 0.03)')
            pr_param_str = "\nNumber of Pruned Parameters: %.1fM"%(pr_params/1e6)
            print(pr_param_str)

            # Evaluate dataset
            acc, avg_inf_time = eval_cifar(model_sp_new, cifar_testloader, device)
            print(f'\nAccuracy - {acc}')
            print(f'Average Inference Time - {avg_inf_time}')
            print('\n*********************************************\n')
