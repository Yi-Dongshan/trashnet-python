import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from model import CustomModel
from DataLoader import DataLoader as CustomDataLoader

# 参数设置
class Config:
    trainList = "data/one-indexed-files-notrash_train.txt"
    valList = "data/one-indexed-files-notrash_val.txt"
    testList = "data/one-indexed-files-notrash_test.txt"
    numClasses = 5
    inputHeight = 384
    inputWidth = 384
    scaledHeight = 256
    scaledWidth = 256
    numChannels = 3
    batchSize = 32
    dataFolder = "data/pics"
    numEpochs = 100
    learningRate = 1.25e-5
    lrDecayFactor = 0.9
    lrDecayEvery = 20
    weightDecay = 2.5e-2
    weightInitializationMethod = "kaiming"
    printEvery = 1
    checkpointEvery = 20
    checkpointName = "checkpoints/checkpoint"
    cuda = True
    gpu = 0
    scale = 1

opt = Config()

# 设置设备
device = torch.device("cuda" if opt.cuda and torch.cuda.is_available() else "cpu")

# 初始化数据加载器
print("Initializing DataLoader")
loader = CustomDataLoader(vars(opt))

# 初始化模型和损失函数
print("Initializing model and criterion")
model = CustomModel(vars(opt)).to(device)
criterion = nn.CrossEntropyLoss().to(device)

# 初始化优化器
optimizer = optim.Adam(model.parameters(), lr=opt.learningRate, weight_decay=opt.weightDecay)

# 训练函数
def train(model):
    print(f"Starting training for {opt.numEpochs} epochs")
    model.train()

    for epoch in range(opt.numEpochs):
        epoch_loss = 0
        for i, batch in enumerate(loader.nextBatch("train", True)):
            data, labels = batch['data'].to(device), batch['labels'].to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % opt.printEvery == 0:
            print(f"Epoch [{epoch+1}/{opt.numEpochs}], Loss: {epoch_loss/len(loader.splits['train']['filePaths']):.4f}")

        if (epoch + 1) % opt.checkpointEvery == 0 or (epoch + 1) == opt.numEpochs:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch
            }
            torch.save(checkpoint, f"{opt.checkpointName}_{epoch+1}.pth")
            print(f"Checkpoint saved at epoch {epoch+1}")

# 测试函数
def test(model, split):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader.nextBatch(split, False):
            data, labels = batch['data'].to(device), batch['labels'].to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the {split} images: {accuracy:.2f}%')
    return accuracy

# 运行训练和测试
train(model)
train_accuracy = test(model, "train")
val_accuracy = test(model, "val")
test_accuracy = test(model, "test")

print(f"Final accuracy on the train set: {train_accuracy:.2f}%")
print(f"Final accuracy on the val set: {val_accuracy:.2f}%")
print(f"Final accuracy on the test set: {test_accuracy:.2f}%")