import torch
from einops import repeat
from flash_attn.ops.triton.layer_norm import RMSNorm as fRMSNorm

import torch
import torch.nn as nn


n_base = 4
state_len = 3

idx = torch.cat([
    torch.arange(n_base**(state_len))[:, None],
    torch.arange(n_base**(state_len))
    .repeat_interleave(n_base).reshape(n_base, -1).T
], dim=1)
#idx = torch.arange(n_base**(state_len)).repeat_interleave(n_base).reshape(n_base, -1).T
print(idx)