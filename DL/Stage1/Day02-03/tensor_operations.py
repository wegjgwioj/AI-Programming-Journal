import torch as th
import tensorflow as tf 
import numpy as np
# 学习张量操作（创建、切片、广播），对比NumPy实现

# 创建张量
myth = th.tensor([1, 2, 3, 4, 5], dtype=th.float32, requires_grad=True)
mynp = np.array([1, 2, 3, 4, 5])

# 切片：从张量或数组中提取子集
print(myth[1:3])  # 输出: tensor([2., 3.], grad_fn=<SliceBackward>)
print(mynp[1:3])  # 输出: [2 3]

# 广播：对张量或数组的每个元素进行操作
print(myth + 1)  # 输出: tensor([2., 3., 4., 5., 6.], grad_fn=<AddBackward0>)
print(mynp + 1)  # 输出: [2 3 4 5 6]

print(myth * 2)  # 输出: tensor([ 2.,  4.,  6.,  8., 10.], grad_fn=<MulBackward0>)
print(mynp * 2)  # 输出: [ 2  4  6  8 10]

print(myth ** 2)  # 输出: tensor([ 1.,  4.,  9., 16., 25.], grad_fn=<PowBackward0>)
print(mynp ** 2)  # 输出: [ 1  4  9 16 25]

# 自动微分：计算张量的梯度
output = myth.sum()  # 对张量求和
output.backward()  # 反向传播计算梯度
print("梯度：", myth.grad)  # 输出: tensor([1., 1., 1., 1., 1.])

