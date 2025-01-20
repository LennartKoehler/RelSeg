import torch
from einops import repeat
from flash_attn.ops.triton.layer_norm import RMSNorm as fRMSNorm

import torch
import torch.nn as nn


    


tensor = torch.rand((2,4), requires_grad=True)
tensorb = tensor * 2
a = tensorb[0,:]
b = tensorb[1,:] * 2
c = a + b
c[0].backward()
print(tensor)
print(tensor.grad)