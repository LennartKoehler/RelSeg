import torch

x = torch.tensor([0,1,2,3], dtype=torch.float, requires_grad=True)

y = torch.tensor([1])

z = x*y
z = z.detach()
z.backward()
print(x.grad)