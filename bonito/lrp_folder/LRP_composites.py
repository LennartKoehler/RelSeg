import operator
import torch
import torch.nn as nn
import torch.nn.functional as F


from lxt.core import Composite
import lxt.functional as lf
import lxt.modules as lm
import lxt.rules as rules


from zennit.composites import LayerMapComposite
import zennit.rules as z_rules
from zennit.image import gridify

#####################
### LRP Composite ###
#####################

class AttentionValueMatmul(nn.Module):
    def forward(self, attn, value):
        return torch.matmul(attn, value)

class ProjSwigluMultiplication(nn.Module):
    def forward(self, a, b):
        return a * b

# must be traced because of the functions, e.g. operator.add!
lxt_comp = Composite({

        nn.SiLU: rules.IdentityRule,
        nn.GELU: rules.IdentityRule,
        nn.Tanh: rules.IdentityRule,
        nn.Sigmoid: rules.IdentityRule,

        ProjSwigluMultiplication: rules.UniformRule,
        AttentionValueMatmul: rules.UniformEpsilonRule,

        nn.Linear: lm.LinearAlphaBeta,
        #nn.Conv1d: rules.EpsilonRule,

        nn.Softmax: lm.SoftmaxDT,
        F.glu: lf.linear_alpha_beta,
        operator.add: lf.add2,
        torch.add: lf.add2,
        operator.mul: lf.mul2,
        operator.truediv: lf.div2,
        operator.matmul: lf.matmul,
        torch.matmul: lf.matmul,
        F.normalize: lf.normalize,
    })

conv_gamma = 0.01

zennit_comp = LayerMapComposite([
    (nn.Conv1d, z_rules.WSquare(conv_gamma)),
    # (nn.Linear, z_rules.WSquare(conv_gamma))
])