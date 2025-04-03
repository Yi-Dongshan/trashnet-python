# trashnet

这是一个用于垃圾分类的卷积神经网络项目，最初是为斯坦福大学的CS 229机器学习课程的期末项目开发的。我们的论文可以在[这里](https://cs229.stanford.edu/proj2016/report/ThungYang-ClassificationOfTrashForRecyclabilityStatus-report.pdf)找到。我们在项目结束后继续改进模型，通过使用Kaiming方法进行权重初始化，测试准确率达到了约75%（使用70/13/17的训练/验证/测试数据划分）。

## 数据集

本项目包含我们收集的数据集，涵盖六个类别：玻璃、纸张、纸板、塑料、金属和垃圾。目前数据集包含2527张图片：
- 501张玻璃
- 594张纸张
- 403张纸板
- 482张塑料
- 410张金属
- 137张垃圾

图片是在白色海报板上拍摄的，使用了自然光和室内光。图片已被调整为512 x 384的大小。原始数据集大小约为3.5GB，超过了git-lfs的最大限制，因此已上传到Google Drive。如果计划使用Python代码预处理原始数据集，请从以下链接下载`dataset-original.zip`并将解压后的文件夹放入`data`文件夹中。

**如果使用此数据集，请引用本仓库。数据集可以从[这里](https://huggingface.co/datasets/garythung/trashnet)下载。**



## 使用

### 第一步：准备数据

解压`data/dataset-resized.zip`。

如果添加更多数据，新文件必须正确编号并放入`data/dataset-original`中的相应文件夹，然后进行预处理。预处理数据涉及删除`data/dataset-resized`文件夹，然后从`trashnet/data`调用`python resize.py`。这将花费大约半小时。

### 第二步：训练模型

使用`train.py`来训练模型。确保数据已准备好并且所有依赖项已安装。

### 第三步：测试模型

使用`test.py`来测试模型。可以选择在训练、验证或测试集上进行测试。

### 第四步：查看结果

使用`plot.py`来查看训练和测试结果。结果将以图形形式保存到指定的输出目录中。

## 贡献

1. Fork本项目！
2. 创建你的功能分支：`git checkout -b my-new-feature`
3. 提交你的更改：`git commit -m 'Add some feature'`
4. 推送到分支：`git push origin my-new-feature`
5. 提交一个pull request

## 致谢

- 感谢斯坦福CS 229秋季2016-2017课程的教学团队！
- 感谢[@e-lab](http://github.com/e-lab)提供的[weight-init Torch模块](http://github.com/e-lab/torch-toolbox/blob/master/Weight-init/weight-init.lua)

## TODOs

- 添加混淆矩阵数据的保存和图形创建
- 重写数据预处理以仅重新处理新图像如果尺寸未更改
