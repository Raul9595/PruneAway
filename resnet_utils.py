
'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import smdebug.pytorch as smd

import os
import argparse

from models import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

best_acc = 0  # best test accuracy

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


# Training
def train(model, trainloader, epoch, model_ext, criterion, optimizer, hook):
    # Starting training job
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        # for named_module in model.named_modules():
    
        #     layer_name = named_module[0]
        #     layer = named_module[1]

        #     if isinstance(layer,  torch.nn.modules.batchnorm.BatchNorm2d) and ((batch_idx%100 == 0) or (batch_idx == 0)):
        #         hook.save_tensor(layer_name + ".running_mean_output_0", layer.running_mean)
        #         hook.save_tensor(layer_name + ".running_var_output_0", layer.running_var)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total), end='\r', flush=True)
        # hook.close()
    print()


def test(model, testloader, epoch, criterion, model_ext):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total), end='\r', flush=True)

    print()
    # Save checkpoint.
    # acc = 100.*correct/total
    # if acc > best_acc:
    #     print('Saving..')
    #     state = {
    #         'net': model.state_dict(),
    #         'acc': acc,
    #         'epoch': epoch,
    #     }
    #     if not os.path.isdir('weights'):
    #         os.mkdir('weights')
    #     torch.save(state, './weights/' + model_ext + '.pth')
    #     best_acc = acc


def start_training(model, trainloader, testloader, model_ext):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.05,
                        momentum=0, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # Registering Job
    job_name = model_ext
    hook = smd.Hook(out_dir=f'./smdebug/{job_name}',
                save_config=smd.SaveConfig(save_interval=100),
                include_collections=['weights', 'gradients', 'biases'])

    hook.register_module(model)
    hook.register_loss(criterion)

    for epoch in range(0, 5):
        train(model, trainloader, epoch, model_ext, criterion, optimizer, hook)
        test(model, testloader, epoch, criterion, model_ext)
        scheduler.step()