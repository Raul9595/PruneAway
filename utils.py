import torch

from models import *


def load_model(device, model_type='resnet50'):
    if model_type == 'resnet50':
         # Loading model
        model = ResNet50()
        model = model.to(device)
        model = torch.nn.DataParallel(model)
        
        model.load_state_dict(torch.load('./weights/resnet50.pth', map_location=device)['net'])
    
    elif model_type == 'resnet101':
        # Loading model
        model = ResNet101()
        model = model.to(device)
        model = torch.nn.DataParallel(model)
        
        model.load_state_dict(torch.load('./weights/resnet101.pth', map_location=device)['net'])
    
    elif model_type == 'resnet152':
         # Loading model
        model = ResNet152()
        model = model.to(device)
        model = torch.nn.DataParallel(model)
        
        model.load_state_dict(torch.load('./weights/resnet152.pth', map_location=device)['net'])
    else:
        print('Invalid model!')
        return None
    
    return model