
import torch

x = torch.randn(5,3,dtype=torch.float)
print(x)

y = x.view(-1,5)   #变形
print(y)

s = torch.empty(5,5)  # 没有进行初始化
print(s)

h = torch.zeros(5,3)
print(h)

print(h.size())   # torch.Size 返回的这个是一个元组

# 加法
a = x+h
print(a)


print(torch.add(x,h))

torch.add(x,h,out=h)
print(h)