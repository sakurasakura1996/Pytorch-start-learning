
import torch
# 将一个Torch张量转换为一个NumPy数组是轻而易举的事情，反之亦然。
#
# Torch张量和NumPy数组将共享它们的底层内存位置，因此当一个改变时,另外也会改变。
a = torch.ones(5)
print(a)

b = a.numpy()
print(b)
# 看numpy数组是如何改变里面的值的
a.add_(1)
print(a)
print(b)

import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a,1,out=a)
print(a)
print(b)
# CPU上的所有张量(CharTensor除外)都支持与Numpy的相互转换。

# CUDA上的张量
# 张量可以使用.to方法移动到任何设备（device）上：
x = torch.randn(4, 4)
if torch.cuda.is_available():
    device = torch.device("cuda")
    y = torch.ones_like(x, device=device)
    x = x.to(device)
    z = x+y
    print(z)
    print(z.to("cpu", torch.double))
else:
    print("没有cuda")




# 没有cuda 你搞尼玛的深度学习啊
# device = torch.device("cuda")
# x = torch.ones(5,3)
# y = x.to(device)
# print(x)
# print(y)
