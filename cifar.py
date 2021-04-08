import glob
import time

import torch
import torchvision
from torchvision import transforms


def time_synchronized():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def load_cifar(mode='train'):
    # Transforming data
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if mode == 'train':
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)

        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=50, shuffle=False, num_workers=2)

        return trainloader
    else:
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)

        testloader = torch.utils.data.DataLoader(
            testset, batch_size=50, shuffle=False, num_workers=2)
        
        return testloader


def eval_cifar(model, dataloader, device):
    correct = 0
    total = 0
    t = 0.
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            print(f'Completed {batch_idx+1} of {len(dataloader)} batches', end='\r', flush=True)
            inputs, targets = inputs.to(device), targets.to(device)

            t0 = time_synchronized()
            outputs = model(inputs)
            t += time_synchronized() - t0

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # Accuracy
    acc = 100.*correct/total
    t = t/10000.
    return acc, t