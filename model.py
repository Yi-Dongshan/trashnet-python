import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomModel(nn.Module):
    def __init__(self, kwargs):
        super(CustomModel, self).__init__()
        num_classes = kwargs.get('numClasses')
        num_channels = kwargs.get('numChannels')
        scale = kwargs.get('scale')

        self.cnn = {
            'conv1Channels': int(96 * scale),
            'conv1Kernel': 11,
            'conv1Stride': 4,
            'conv1Pad': 2,
            'pool1Kernel': 3,
            'pool1Stride': 2,
            'pool1Pad': 0,

            'conv2Channels': int(256 * scale),
            'conv2Kernel': 5,
            'conv2Stride': 1,
            'conv2Pad': 2,
            'pool2Kernel': 3,
            'pool2Stride': 2,
            'pool2Pad': 0,

            'conv3Channels': int(384 * scale),
            'conv3Kernel': 3,
            'conv3Stride': 1,
            'conv3Pad': 1,

            'conv4Channels': int(384 * scale),
            'conv4Kernel': 3,
            'conv4Stride': 1,
            'conv4Pad': 1,

            'conv5Channels': int(256 * scale),
            'conv5Kernel': 3,
            'conv5Stride': 1,
            'conv5Pad': 1,
            'pool5Kernel': 3,
            'pool5Stride': 2,
            'pool5Pad': 0,

            'fc6Channels': int(4096 * scale),
            'fc7Channels': int(4096 * scale)
        }

        self.features = nn.Sequential(
            nn.Conv2d(num_channels, self.cnn['conv1Channels'], kernel_size=self.cnn['conv1Kernel'], stride=self.cnn['conv1Stride'], padding=self.cnn['conv1Pad']),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=self.cnn['pool1Kernel'], stride=self.cnn['pool1Stride'], padding=self.cnn['pool1Pad']),
            nn.Conv2d(self.cnn['conv1Channels'], self.cnn['conv2Channels'], kernel_size=self.cnn['conv2Kernel'], stride=self.cnn['conv2Stride'], padding=self.cnn['conv2Pad']),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=self.cnn['pool2Kernel'], stride=self.cnn['pool2Stride'], padding=self.cnn['pool2Pad']),
            nn.Conv2d(self.cnn['conv2Channels'], self.cnn['conv3Channels'], kernel_size=self.cnn['conv3Kernel'], stride=self.cnn['conv3Stride'], padding=self.cnn['conv3Pad']),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.cnn['conv3Channels'], self.cnn['conv4Channels'], kernel_size=self.cnn['conv4Kernel'], stride=self.cnn['conv4Stride'], padding=self.cnn['conv4Pad']),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.cnn['conv4Channels'], self.cnn['conv5Channels'], kernel_size=self.cnn['conv5Kernel'], stride=self.cnn['conv5Stride'], padding=self.cnn['conv5Pad']),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=self.cnn['pool5Kernel'], stride=self.cnn['pool5Stride'], padding=self.cnn['pool5Pad'])
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.cnn['conv5Channels'] * 7 * 7, self.cnn['fc6Channels']),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(self.cnn['fc6Channels'], self.cnn['fc7Channels']),
            nn.ReLU(inplace=True),
            nn.Linear(self.cnn['fc7Channels'], num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x