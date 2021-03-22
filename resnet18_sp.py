import torch
import torchvision
from torchvision.models import resnet18
from pruning import prune_model
import torch_pruning as tp
import numpy as np
from eval_imagenet import eval_data



if __name__ == '__main__':
    cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')

    #Original model
    model = resnet18(pretrained=True)
    org_params = sum([np.prod(p.size()) for p in model.parameters()])

    org_param_str = "Number of Original Parameters: %.1fM"%(org_params/1e6)
    print(org_param_str)

    print('Evaluating pruned model')
    eval_data(model, './results/resnet18_org.txt', org_param_str)


    # Pruning model (Structured)
    prune_model(model)
    pr_params = sum([np.prod(p.size()) for p in model.parameters()])

    pr_param_str = "Number of Pruned Parameters (Structured): %.1fM"%(pr_params/1e6)
    print(pr_param_str)

    print('Evaluating pruned model (Structured)')
    eval_data(model, './results/resnet18_sp.txt', pr_param_str)