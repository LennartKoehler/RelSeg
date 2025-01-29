import torch
from einops import repeat
from flash_attn.ops.triton.layer_norm import RMSNorm as fRMSNorm

import torch
import torch.nn as nn


    

torch.manual_seed(1)
x = torch.rand((2,3,4,5), dtype=torch.float)

torch.manual_seed(1)
y = torch.rand((2,3,4,5), dtype=torch.float)

pad_x = torch.nn.functional.pad(x, (1,0,0,0,0,0,0,0), value=0)

pad_y = torch.nn.functional.pad(y, (1,0), value=0)

print(pad_x, "\n", pad_y)
print(torch.equal(pad_x, pad_y))
