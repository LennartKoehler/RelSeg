import torch

x = torch.tensor([[0,1,2,3],[5,6,7,8]], dtype=torch.float, requires_grad=True)

b = torch.tensor([[0,1,0,1],[1,1,1,1]])
x = x[b!=0]
print(x)