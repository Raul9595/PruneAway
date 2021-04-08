import os
import shutil
import argparse
import numpy as np
import copy
from datetime import datetime
import torch
import torchvision
import torch_pruning as tp
from smdebug.trials import create_trial

from cifar import load_cifar, eval_cifar
from models import *
from utils import load_model
from pruning import compute_filter_ranks, normalize_filter_ranks, get_smallest_filters, prune_model_iterative
from resnet_utils import start_training


if __name__ == '__main__':
    
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)

    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', type=str, default='cifar', help='Evaluation dataset')
    parser.add_argument('--model', type=str, default='resnet152', help='Evaluation dataset')
    opt = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = load_model(device, opt.model)

    if model is None:
        pass

    else:
        model_ip = copy.deepcopy(model)
        
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

            # Pruning model (Iterative)
            for pruning_step in range(10):

                if os.path.exists('smdebug/' + opt.model + '_ip'):
                    shutil.rmtree('smdebug/' + opt.model + '_ip')

                model_ip_new = copy.deepcopy(model_ip)

                start_training(model_ip_new, cifar_trainloader, cifar_testloader, opt.model + '_ip')

                smdebug_trial = create_trial('./smdebug/' + opt.model + '_ip')
                activation_outputs = []
                gradients = []
                step_list = []

                for step in smdebug_trial.steps():
                    try:
                        _ = smdebug_trial.tensor('gradient/DataParallel_module.layer1.0.conv1.weight').value(step)
                        step_list.append(step)
                    except:
                        continue

                for tensor_name in smdebug_trial.tensor_names():
                    if '.weight' in tensor_name and 'gradient/' not in tensor_name and 'layer' in tensor_name and ('conv1' in tensor_name or 'conv2' in tensor_name):
                        activation_outputs.append(tensor_name)
                        gradients.append('gradient/' + tensor_name)

                filters = compute_filter_ranks(smdebug_trial, activation_outputs, gradients, step_list)
                filters = normalize_filter_ranks(filters)
                filters_list = get_smallest_filters(filters, 1600)

                step = smdebug_trial.steps()[-1]

                model_ip = prune_model_iterative(model_ip, filters_list, smdebug_trial, step)

                # save pruned model
                checkpoint = {'model': model_ip,
                            'state_dict': model_ip.state_dict()}
                torch.save(checkpoint, './weights/' + opt.model + '_ip.pth')

                # clean up
                del model_ip

                checkpoint = torch.load('./weights/' + opt.model + '_ip.pth', map_location=device)

                model_ip = checkpoint['model']
                model_ip = model_ip.to(device)
                model_ip.load_state_dict(checkpoint['state_dict'])

                # Pruned parameters
                pr_params = sum([np.prod(p.size()) for p in model_ip.parameters()])

                print('Evaluating pruned model (Iterative) at step ' + str(pruning_step))
                pr_param_str = "\nNumber of Pruned Parameters: %.1fM"%(pr_params/1e6)
                print(pr_param_str)

                # Evaluate dataset
                acc, avg_inf_time = eval_cifar(model_ip, cifar_testloader, device)
                print(f'\nAccuracy - {acc}')
                print(f'Average Inference Time - {avg_inf_time}')
                print('\n*********************************************\n')

                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                print("Current Time =", current_time)