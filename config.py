import torch
import torch.nn as nn
import torch.nn.functional as F 


class ResidualBlock(nn.Module):
    '''
    Implements a single Residual Block for a ResNet Architecture. Each block consists of a convolutional layer, followed by 
    batch normalization and ReLU activation. Optional Downsampling layer can be applied to the residual connection to match 
    dimensions. The residual connection is added to the output of the second convolutional layer before applying the final
    ReLU activation. 
    '''
    def __init__(self, in_channels, out_channels, stride=1, downsample=None): 
        #Stride hyperparameter is included because it is not the same throughout the network
        super().__init__()
        #First convolutional layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride,padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
        )
        #Second convolutional layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
        )
        self.relu = nn.ReLU() #Final activation
        self.downsample = downsample #Optional downsampling layer
    
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.downsample:
            residual = self.downsample(residual)
        x += residual
        x = self.relu(x)
        return x
    
class ResNet(nn.Module):
    '''
    Complete Implementation of the ResNet Architecture by combining the multiple residual blocks together to form the 34 layer resnet,
    addressed in the machine learning research paper. The network begins with a single convolution layer, followed by a maxpooling to 
    reduce the dimensions of the original image. Then, 4 residual blocks are added, followed up with an average-pooling then one fully
    connected layer into a 10 number class softmax. 
    '''
    def __init__(self, block, layers, num_classes=10):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3), #These values need checking
            nn.BatchNorm2d(num_features=64),
            nn.ReLU()
        )
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) #These values need checking

        self.block0 = self._make_layer(block, 64, layers[0], stride=1) #These stride values need checking
        self.block1 = self._make_layer(block, 128, layers[1], stride=2)
        self.block2 = self._make_layer(block, 256, layers[2], stride=2)
        self.block3 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1) #Written on the paper but need to discover what it is
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels

        for i in range(1, blocks):
            layers.append(block(self.in_channels, out_channels)) #Need to discover why the stride is not set to anything else here (why not =2)
        
        return nn.Sequential(*layers)
    
    def forward (self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x