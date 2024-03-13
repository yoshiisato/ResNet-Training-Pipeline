'''
This script implements the training process for a deep learning model, focusing on the Residual Network (ResNet) architecture. It includes:

    - Data Preprocessing: Preparation and augmentation of the dataset for optimal training performance.
    - Training Loop: The core loop where the model is trained over several epochs, with gradients computed and weights updated.
    - Validation: Evaluation of the model's performance on a separate validation set to monitor overfitting and guide hyperparameter tuning.
'''

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch.utils.data import SubsetRandomSampler
import torch.nn as nn
import numpy as np
from config import ResNet, ResidualBlock

def data_loader(data_dir, batch_size, random_seed=42, validation_size=0.1, shuffle=True, test=False):
    #The mean and standard deviation of RGB values in the CIFAR10 dataset, leads to faster convergence and training
    normalize = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010],
        )

    #Defining Transforms
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize,
    ])

    if test:
        dataset = CIFAR10(root=data_dir, train=False, transform=transform, download=True)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return dataloader
    
    trainset = CIFAR10(root=data_dir, train=True, transform=transform, download=True)
    validset = CIFAR10(root=data_dir, train=True, transform=transform, download=True)

    num_train = len(trainset)
    indices = list(range(num_train))
    split = int(np.floor(validation_size*num_train))

    #Indices to split training dataset into validation and training subsets
    train_index, valid_index = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_index)
    valid_sampler = SubsetRandomSampler(valid_index)

    trainloader = DataLoader(trainset, batch_size=batch_size, sampler=train_sampler)
    validloader = DataLoader(validset, batch_size=batch_size, sampler=valid_sampler)
    
    return (trainloader, validloader)

trainloader, validloader = data_loader(data_dir='./data', batch_size=64)
testloader = data_loader(data_dir='./data', batch_size=64, test=True)

net = ResNet(ResidualBlock, [3,4,6,3]) #add .to('cuda') for GPU on google colab

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, weight_decay=0.001, momentum=0.9)

total_step = len(trainloader)

import gc
num_epochs = 20
batch_size = 16

for epoch in range(num_epochs):
    for batch in trainloader:
        images, labels = batch
        #Optionally move images and labels to GPU

        #Forward pass
        outputs = net(images)
        loss = criterion(outputs, labels)

        #Backpropagtion and Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # del images, labels, outputs
        # torch.cuda.empty_cache()
        # gc.collect()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# Validation
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in validloader:
            # images = images.to(device)
            # labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del images, labels, outputs
    
        print('Accuracy of the network on the {} validation images: {} %'.format(5000, 100 * correct / total)) 