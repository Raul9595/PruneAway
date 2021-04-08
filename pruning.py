import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch_pruning as tp

from models import *


def prune_model_structured(model, device, amt):
    model.cpu()
    dummy_input =  torch.randn(1, 3, 32, 32)

    DG = tp.DependencyGraph().build_dependency(model.module, dummy_input)

    def prune_conv(conv, amount=0.015):
        strategy = tp.strategy.L1Strategy()
        pruning_index = strategy(conv.weight, amount=amount)
        plan = DG.get_pruning_plan(conv, tp.prune_conv, pruning_index)
        plan.exec()
    
    block_id = 0
    for m in model.module.modules():
        if isinstance( m, Bottleneck):
            try:
                prune_conv(m.conv1, amt)
                prune_conv(m.conv2, amt)
                prune_conv(m.conv3, amt)
            except:
                continue
            block_id+=1
    return model.to(device)


def compute_filter_ranks(smdebug_trial, activation_outputs, gradients, step_list):
    filters = {}
    for activation_output_name, gradient_name in zip(activation_outputs, gradients):
        for step in step_list:

            activation_output = smdebug_trial.tensor(activation_output_name).value(step)
            gradient = smdebug_trial.tensor(gradient_name).value(step)
            rank = activation_output * gradient
            rank = np.mean(rank, axis=(1,2,3))

            if activation_output_name not in filters:
                filters[activation_output_name] = 0
            filters[activation_output_name] += rank
    return filters


def normalize_filter_ranks(filters):
    for activation_output_name in filters:
        rank = np.abs(filters[activation_output_name])
        rank = rank / np.sqrt(np.sum(rank * rank))
        filters[activation_output_name] = rank
    return filters


def get_smallest_filters(filters, n):
    filters_list = []
    for layer_name in sorted(filters.keys()):
        for channel in range(filters[layer_name].shape[0]):
            filters_list.append((layer_name, channel, filters[layer_name][channel], ))

    filters_list.sort(key = lambda x: x[2])
    filters_list = filters_list[:n]
    # print("The", n, "smallest filters", filters_list)

    return filters_list


def prune_model_iterative(model, filters_list, trial, step):
    
    # dict that has a list of filters to be pruned per layer
    filters_dict = {}
    for layer_name, channel,_  in filters_list:
        if layer_name not in filters_dict:
            filters_dict[layer_name] = []
        filters_dict[layer_name].append(channel)
    
    counter = 0
    in_channels_dense = 0
    exclude_filters = None
    in_channels = 3
    exclude = False
    save_for_shortcut = None

    #iterate over layers in the ResNet model
    for named_module in model.named_modules():
        
        # for named_module1 in model.named_modules():
        #     if named_module1[0] == 'module.layer1.1.bn1':
        #         print(named_module1[1].weight.shape)
    
        layer_name = named_module[0]
        layer = named_module[1]
        
        #check if current layer is a convolutional layer
        if isinstance(layer, torch.nn.modules.conv.Conv2d) and 'shortcut' not in layer_name:
            #remember the output channels of non-pruned convolution (needed for pruning first fc layer)
            in_channels_dense = layer.out_channels

            #create key to find right weights/bias/filters for the corresponding layer
            weight_name = "DataParallel_" + layer_name + ".weight"
        
            #get weight values from last available training step
            weight = trial.tensor(weight_name).value(step)
            
            #we need to adjust the number of input channels,
            #if previous covolution has been pruned
            # print( "current:", layer.in_channels, "previous", in_channels, layer_name, weight_name)
            # if 'conv1' in layer_name or 'conv2' in layer_name or 'conv3' in layer_name:
            if layer.in_channels != in_channels:
                layer.in_channels = in_channels
                weight  = np.delete(weight, exclude_filters, axis=1)
                if 'conv1' in layer_name:
                    save_for_shortcut = exclude_filters
                exclude_filters = None
            else:
                if 'conv1' in layer_name:
                    save_for_shortcut = None
                    
            #if current layer is in the list of filters to be pruned
            layer_id = 'DataParallel_' + layer_name + '.weight'

            try:
                # print("Reduce output channels for conv layer",  layer_id, "from",  layer.out_channels, "to", layer.out_channels - len(filters_dict[layer_id]))

                #set new output channels
                layer.out_channels = layer.out_channels - len(filters_dict[layer_id]) 

                #remove corresponding filters from weights and bias
                #convolution weights have dimension: filter x channel x kernel x kernel
                exclude_filters = filters_dict[layer_id]

                weight  = np.delete(weight, exclude_filters, axis=0)
            except:
                pass
                              
            #remember new size of output channels, because we need to prune subsequent convolution
            in_channels = layer.out_channels  

            #set pruned weight and bias
            layer.weight.data = torch.from_numpy(weight)
        
        if isinstance(layer, torch.nn.modules.conv.Conv2d) and 'shortcut' in layer_name:
            #create key to find right weights/bias/filters for the corresponding layer
            weight_name = "DataParallel_" + layer_name + ".weight"
        
            #get weight values from last available training step
            weight = trial.tensor(weight_name).value(step)
                    
            #if current layer is in the list of filters to be pruned
            layer_id = 'DataParallel_' + layer_name.replace('shortcut.0', 'conv3') + '.weight'

            try:
                # print("Reduce output channels for conv layer",  layer_id, "from",  layer.out_channels, "to", layer.out_channels - len(filters_dict[layer_id]))

                #set new output channels
                layer.out_channels = layer.out_channels - len(filters_dict[layer_id]) 

                #remove corresponding filters from weights and bias
                #convolution weights have dimension: filter x channel x kernel x kernel
                exclude_filters = filters_dict[layer_id]
                weight  = np.delete(weight, exclude_filters, axis=0)

                layer.in_channels = layer.in_channels - len(save_for_shortcut)
                weight  = np.delete(weight, save_for_shortcut, axis=1)
            except:
                pass
                              
            #remember new size of output channels, because we need to prune subsequent convolution
            in_channels = layer.out_channels

            #set pruned weight and bias
            layer.weight.data = torch.from_numpy(weight)

            #remember the output channels of non-pruned convolution (needed for pruning first fc layer)
            in_channels_dense = layer.out_channels
            
        if isinstance(layer,  torch.nn.modules.batchnorm.BatchNorm2d):

            #get weight values from last available training step
            weight_name = "DataParallel_" + layer_name + ".weight"
            weight = trial.tensor(weight_name).value(step)
            
            #get bias values from last available training step
            bias_name = "DataParallel_" + layer_name + ".bias"
            bias = trial.tensor(bias_name).value(step)
            
            #get running_mean values from last available training step
            # mean_name = layer_name + ".running_mean_output_0"
            # mean = trial.tensor(mean_name).value(step)
            mean = layer.running_mean.detach().cpu().numpy()
            
            #get running_var values from last available training step
            # var_name = layer_name + ".running_var_output_0"
            # var = trial.tensor(var_name).value(step)
            var = layer.running_var.detach().cpu().numpy()

            if 'shortcut' not in layer_name:
                layer_id = 'DataParallel_' + layer_name.replace('bn', 'conv') + '.weight'
            else:
                layer_id = 'DataParallel_' + layer_name.replace('shortcut.1', 'conv3') + '.weight'

            try:
                # print("Reduce bn layer",  layer_id, "from",  weight.shape[0], "to", weight.shape[0] - len(filters_dict[layer_id]))

                #remove corresponding filters from weights and bias
                #convolution weights have dimension: filter x channel x kernel x kernel
                exclude_filters = filters_dict[layer_id]
                weight  = np.delete(weight, exclude_filters, axis=0)
                bias =  np.delete(bias, exclude_filters, axis=0)
                mean =  np.delete(mean, exclude_filters, axis=0)
                var  =  np.delete(var, exclude_filters, axis=0)
            except:
                pass

            #set pruned weight and bias
            layer.weight.data = torch.from_numpy(weight)
            layer.bias.data = torch.from_numpy(bias)
            layer.running_mean.data = torch.from_numpy(mean)
            layer.running_var.data = torch.from_numpy(var)
            layer.num_features = weight.shape[0]
            in_channels = weight.shape[0]
            
        if isinstance(layer, torch.nn.modules.linear.Linear):

            #get weight values from last available training step
            weight_name = "DataParallel_" + layer_name + ".weight"
            weight = trial.tensor(weight_name).value(step)
            
            #get bias values from last available training step
            bias_name = "DataParallel_" + layer_name + ".bias"
            bias = trial.tensor(bias_name).value(step)
            
            #prune first fc layer
            if exclude_filters is not None:
                 #in_channels_dense is the number of output channels of last non-pruned convolution layer
                params = int(layer.in_features/in_channels_dense)

                #prune weights of first fc layer
                indexes = []
                for i in exclude_filters: 
                    indexes.extend(np.arange(i * params, (i+1)*params))
                    if indexes[-1] > weight.shape[1]:
                        indexes.extend(np.arange(weight.shape[1] - params , weight.shape[1]))   
                weight  = np.delete(weight, indexes, axis=1)         

                # print("Reduce weights for first linear layer from", layer.in_features, "to", weight.shape[1])
                 #set new in_features
                layer.in_features = weight.shape[1]
                exclude_filters = None

            #set weights
            layer.weight.data = torch.from_numpy(weight)

            #set bias
            layer.bias.data = torch.from_numpy(bias)
  
    return model