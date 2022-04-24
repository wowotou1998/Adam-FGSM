import torch


def change_tensor_1(t):
    t[0][0] = 1
    t[1] = torch.tensor([3, 3])
    t0 = t * 9
    t2 = torch.flatten(t)
    t3 = t2.reshape((-1, 1))
    t4 = t[1:2]

    print(id(t), id(t0), id(t1), id(t2), id(t3), id(t4))


t1 = torch.ones((2, 2))
t8 = torch.ones((2)) * 2
t1[0][0] = 3
t1[0][1] = 4

print('before put into function t1 id', id(t1), )
change_tensor_1(t1)
print('after put into function t1 ', t1)
print(torch.__version__)
print('t1', t1)
print(torch.linalg.norm(t1, ord=1, dim=1))
print(t1 / t8)

x = torch.Tensor([1, 2])
print(x, x.shape)
x = x.unsqueeze(1)
print(x, x.shape)

x = x.expand(2, 6)
print(x, x.shape)
x = x.view(2, 3, 2)
print(x, x.shape)


def g():
    return [0, 1]


a, b = g()
print(a, b)
y = torch.ones(6, dtype=torch.float) * 6
print(y / 3)
a = torch.zeros((2, 1))
b = torch.nonzero(a)
c = torch.index_select(a, dim=0,index=torch.flatten(b))
print(type(c))
print(type(c.shape[0]))
print(c)
