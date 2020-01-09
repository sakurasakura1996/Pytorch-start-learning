# pytorch 自动求导
"""
PyTorch中，所有神经网络的核心是 autograd 包。先简单介绍一下这个包，然后训练我们的第一个的神经网络。

autograd 包为张量上的所有操作提供了自动求导机制。它是一个在运行时定义（define-by-run）的框架，
这意味着反向传播是根据代码如何运行来决定的，并且每次迭代可以是不同的.

torch.Tensor 是这个包的核心类。如果设置它的属性 .requires_grad 为 True，那么它将会追踪对于该张量的所有操作。
当完成计算后可以通过调用 .backward()，来自动计算所有的梯度。这个张量的所有梯度将会自动累加到.grad属性.

要停止 tensor 历史记录的跟踪，您可以调用 .detach()，它将其与计算历史记录分离，并防止将来的计算被跟踪。

要停止跟踪历史记录（和使用内存），您还可以将代码块使用 with torch.no_grad(): 包装起来。在评估模型时，

这是特别有用，因为模型在训练阶段具有 requires_grad = True 的可训练参数有利于调参，但在评估阶段我们不需要梯度。

还有一个类对于 autograd 实现非常重要那就是 Function。Tensor 和 Function 互相连接并构建一个非循环图，
它保存整个完整的计算过程的历史信息。每个张量都有一个 .grad_fn 属性保存着创建了张量的 Function 的引用，
（如果用户自己创建张量，则grad_fn 是 None ）。

如果你想计算导数，你可以调用 Tensor.backward()。如果 Tensor 是标量（即它包含一个元素数据），
则不需要指定任何参数backward()，但是如果它有更多元素，则需要指定一个gradient 参数来指定张量的形状。
"""

import torch

# 创建一个张量，设置requires_grad = True 来跟踪与它相关的计算
x = torch.ones(2,2,requires_grad=True)
print(x)

# 针对张量做一个操作
y = x+2
print(y)

# 上面的y作为操作的结果被创建，所以他有了grad_fn 来保存创建张量的function的引用
print(y.grad_fn)

z = y*y*3
out = z.mean()

print(z,out)

# .requires_grad_( ... ) 会改变张量的 requires_grad 标记。输入的标记默认为 False ，如果没有提供相应的参数。

a = torch.rand(2,2)
a = ((a*3)/(a-1))
print(a.requires_grad)

a.requires_grad_(True)
print(a.requires_grad)

b = (a*a).sum()
print(b.grad_fn)

# 梯度
# 我们现在向后传播，因为输出包含一个标量，out.backward()等同于out.backward(torch.tensor(1.))

out.backward()

print(x.grad)

# 现在让我们看一个雅可比向量积的例子
x = torch.randn(3,requires_grad=True)

y = x*2
while y.data.norm() < 1000:
    y = y*2
print(y)














