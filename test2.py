import torch
from einops import repeat
from flash_attn.ops.triton.layer_norm import RMSNorm as fRMSNorm

import torch
import torch.nn as nn

a = torch.tensor([1,2,3])
b = torch.tensor([4,4])
c = torch.tensor([a,b])
print(c)