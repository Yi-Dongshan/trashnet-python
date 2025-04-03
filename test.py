import torch
import torch.nn as nn
from model import CustomModel
from DataLoader import DataLoader as CustomDataLoader

# 参数设置
class Config:
    checkpoint = "checkpoints/checkpoint_final.pth"
    split = ""  # "train", "val", or "test"
    cuda = True

opt = Config()

# 设置设备
device = torch.device("cuda" if opt.cuda and torch.cuda.is_available() else "cpu")

# 加载模型和损失函数
print("Initializing model")
checkpoint = torch.load(opt.checkpoint, map_location=device)
model = CustomModel(checkpoint['opt']).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
criterion = nn.CrossEntropyLoss().to(device)

# 初始化数据加载器
print("Initializing DataLoader")
loader = CustomDataLoader(checkpoint['opt'])

# 测试函数
def test(model, split):
    assert split in ["train", "val", "test"]
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

# 运行测试
if opt.split == "":
    for split in ["train", "val", "test"]:
        accuracy = test(model, split)
        print(f"Accuracy on the {split} split: {accuracy:.2f}%")
else:
    accuracy = test(model, opt.split)
    print(f"Accuracy on the {opt.split} split: {accuracy:.2f}%")