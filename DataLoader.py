import os
import numpy as np
from PIL import Image

GLASS = 1
PAPER = 2
CARDBOARD = 3
PLASTIC = 4
METAL = 5
TRASH = 6

class DataLoader:
    def __init__(self, kwargs):
        self.splits = {
            'train': {},
            'val': {},
            'test': {}
        }
        self.splits['train']['list'] = kwargs.get('trainList')
        self.splits['test']['list'] = kwargs.get('testList')
        self.splits['val']['list'] = kwargs.get('valList')

        self.opt = {
            'inputHeight': kwargs.get('inputHeight'),
            'inputWidth': kwargs.get('inputWidth'),
            'scaledHeight': kwargs.get('scaledHeight'),
            'scaledWidth': kwargs.get('scaledWidth'),
            'numChannels': kwargs.get('numChannels'),
            'batchSize': kwargs.get('batchSize'),
            'dataFolder': kwargs.get('dataFolder')
        }

        for split in self.splits:
            self.splits[split]['index'] = 0
            self.splits[split]['filePaths'], self.splits[split]['labels'] = self.load_list(self.splits[split]['list'])
            self.splits[split]['count'] = len(self.splits[split]['filePaths'])

        self.meanImage = self.get_mean_training_image(self.splits['train']['filePaths'])

    def next_batch(self, split, augment=False):
        assert split in ['train', 'val', 'test']

        imageData = []
        imageLabels = []

        while len(imageData) < self.opt['batchSize']:
            index = self.splits[split]['index']
            imagePath = self.splits[split]['filePaths'][index]
            imageLabel = self.splits[split]['labels'][index]

            image = Image.open(imagePath).convert('RGB')
            image = image.resize((self.opt['scaledWidth'], self.opt['scaledHeight']))
            image = np.array(image, dtype=np.float32)
            image -= self.meanImage

            if split == 'train' and augment:
                transform = np.random.randint(1, 5)
                if transform == 1:
                    image = self.random_crop(image, self.opt['scaledHeight'] // 20)
                elif transform == 2:
                    image = self.horizontal_flip(image, 0.5)
                elif transform == 3:
                    image = self.add_noise(image, np.random.uniform(-5, 5))

            image = image.transpose((2, 0, 1))  # Change to channel-first
            imageData.append(image)
            imageLabels.append(imageLabel)
            self.splits[split]['index'] += 1
            if self.splits[split]['index'] >= self.splits[split]['count']:
                self.splits[split]['index'] = 0
                break

        batch = {
            'data': np.stack(imageData),
            'labels': np.array(imageLabels)
        }
        return batch

    def load_list(self, fileListPath):
        filePaths = []
        fileLabels = []
        with open(fileListPath, 'r') as file:
            for line in file:
                tokens = line.split()
                filePath, fileLabel = tokens[0], int(tokens[1])
                if fileLabel == GLASS:
                    filePath = os.path.join(self.opt['dataFolder'], 'glass', filePath)
                elif fileLabel == PAPER:
                    filePath = os.path.join(self.opt['dataFolder'], 'paper', filePath)
                elif fileLabel == CARDBOARD:
                    filePath = os.path.join(self.opt['dataFolder'], 'cardboard', filePath)
                elif fileLabel == PLASTIC:
                    filePath = os.path.join(self.opt['dataFolder'], 'plastic', filePath)
                elif fileLabel == METAL:
                    filePath = os.path.join(self.opt['dataFolder'], 'metal', filePath)
                elif fileLabel == TRASH:
                    filePath = os.path.join(self.opt['dataFolder'], 'trash', filePath)

                filePaths.append(filePath)
                fileLabels.append(fileLabel)
        return filePaths, fileLabels

    def get_mean_training_image(self, filePaths):
        means = np.zeros(3)
        numImages = 0

        for filePath in filePaths:
            image = Image.open(filePath).convert('RGB')
            image = image.resize((self.opt['scaledWidth'], self.opt['scaledHeight']))
            image = np.array(image, dtype=np.float32)
            numImages += 1
            for channel in range(3):
                means[channel] += (image[:, :, channel].mean() - means[channel]) / numImages

        meanImage = np.zeros((self.opt['scaledHeight'], self.opt['scaledWidth'], 3))
        for channel in range(3):
            meanImage[:, :, channel] = means[channel]

        return meanImage

    def random_crop(self, image, size):
        h, w, _ = image.shape
        if w == size and h == size:
            return image

        x1, y1 = np.random.randint(0, w - size), np.random.randint(0, h - size)
        image[x1:x1 + size, y1:y1 + size] = 0
        return image

    def horizontal_flip(self, image, prob):
        if np.random.rand() < prob:
            return np.fliplr(image)
        return image

    def add_noise(self, image, augNoise):
        noise = np.random.randn(*image.shape) * augNoise
        return image + noise