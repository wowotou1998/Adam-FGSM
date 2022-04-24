import torch

a = torch.ones((3, 2, 2))
a[0][0][1] = 14
a[0][1][1] = -9
b = torch.clamp(a, min=-1, max=5)
print(b)
