### 迁移学习（ResNet）

- 学习目标：掌握迁移学习的概念，并使用预训练的 ResNet 模型进行图像分类任务。
- 学习内容：
  - 理解迁移学习的基本原理和优势。
  - 学习 ResNet 架构及其变体。
  - 使用 PyTorch 加载预训练的 ResNet 模型，并在 CIFAR10 数据集上进行微调。

### 任务

1. **理解迁移学习的基本原理和优势**

   - 阅读相关文献和教程，理解迁移学习如何通过使用预训练模型来提高训练效率和模型性能。

   ###### 定义：


   * 把一个任务上训练好的模型参数，迁移到另一个相关任务中作为初始值。

   ###### 优势：

   * 节省训练时间（无需从头训练）
   * 小数据集也能获得较好效果（避免过拟合）
   * 复用底层特征提取能力（边缘/纹理等基础特征通用）

   ###### 策略

   * 特征提取（冻结卷积层）：仅训练新添加的全连接层
   * 微调（Fine-tuning）：解冻部分或全部卷积层，用较小学习率继续训练
2. **学习 ResNet 架构及其变体**

   - 阅读 ResNet 论文，了解其网络结构和创新点。

     [ResNet 原论文及原作者讲解_resnet原论文-CSDN博客](https://blog.csdn.net/xiaoyuting999/article/details/135077588)

     [论文原文](DL\Stage2\Day02\论文原文.pdf)
   - 研究 ResNet 的不同变体（如 ResNet-18、ResNet-34、ResNet-50 等），理解它们的区别和应用场景。

     ##### ResNet-18

     ResNet-18 是 ResNet 系列中最浅的网络之一，包含 18 层。它适用于计算资源有限的场景，如移动设备或实时应用。

     ##### ResNet-34

     ResNet-34 是 ResNet-18 的扩展版本，包含 34 层。它在保持较低计算复杂度的同时，提供了更高的准确性，适用于中等计算资源的场景。

     ##### ResNet-50

     ResNet-50 是 ResNet 系列中较深的网络之一，包含 50 层。它通过引入瓶颈结构（bottleneck blocks）来减少计算量，同时提高了模型的表达能力。ResNet-50 常用于需要高精度的任务，如图像分类和目标检测。

     ##### 应用场景


     - **ResNet-18**: 适用于实时应用、移动设备、嵌入式系统等计算资源有限的场景。
     - **ResNet-34**: 适用于中等计算资源的场景，如中型服务器或桌面计算机。
     - **ResNet-50**: 适用于高精度要求的任务，如图像分类、目标检测、语义分割等。
3. **使用 PyTorch 加载预训练的 ResNet 模型**

   - 安装 PyTorch 和 torchvision 库。
   - 编写代码加载预训练的 ResNet 模型，并在 CIFAR10 数据集上进行微调。
4. **评估模型性能**

   - 在测试集上评估模型的准确率。
   - 可视化训练过程中的损失和准确率变化。
