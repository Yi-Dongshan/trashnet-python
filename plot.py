import os
import torch
import matplotlib.pyplot as plt

# 参数设置
class Config:
    checkpoint = "checkpoints/checkpoint11/checkpoint_final.pth"
    outputDir = "checkpoints/checkpoint11"

opt = Config()

# 加载检查点
assert opt.checkpoint, "Need a trained network file to load."
checkpoint = torch.load(opt.checkpoint, map_location='cpu')
trainLossHistory = torch.tensor(checkpoint['trainLossHistory'])
trainAccHistory = torch.tensor(checkpoint['trainAccHistory'])
valLossHistory = torch.tensor(checkpoint['valLossHistory'])
valAccHistory = torch.tensor(checkpoint['valAccHistory'])
epochs = torch.tensor(checkpoint['epochs'])

assert epochs.size(0) == trainLossHistory.size(0), "The number of epochs must correspond to the number of train loss history points."
assert epochs.size(0) == trainAccHistory.size(0), "The number of epochs must correspond to the number of train accuracy history points."
assert epochs.size(0) == valLossHistory.size(0), "The number of epochs must correspond to the number of val loss history points."
assert epochs.size(0) == valAccHistory.size(0), "The number of epochs must correspond to the number of val accuracy history points."

# 绘制训练损失
plt.figure()
plt.plot(epochs.numpy(), trainLossHistory.numpy(), label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(opt.outputDir, 'training-loss.png'))

# 绘制准确率
plt.figure()
plt.plot(epochs.numpy(), trainAccHistory.numpy(), label='Training Accuracy')
plt.plot(epochs.numpy(), valAccHistory.numpy(), label='Validation Accuracy')
plt.title('Accuracy Fitting')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(os.path.join(opt.outputDir, 'accuracy.png'))

plt.show()