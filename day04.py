import torch

x = torch.randn(3, requires_grad=True)

y = x * 2
while y.data.norm() < 1000:
    y = y * 2

print(y)

v = torch.tensor([0.1, 1.0, 1.0], dtype=torch.float)
y.backward(v)  #这个v应该是权重值，这里表达的是 从y向x反向传播得到梯度，通过x.grad来求得梯度值

print(x.grad)