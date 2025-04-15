import torch
import torch.nn as nn
import inspect
import lxt.functional as lf
from lxt.modules import INIT_MODULE_MAPPING, initialize_bias
import torch.fx
from torch.autograd import Function
from lxt.functional import conservation_check_wrap
import torch.nn.functional as F



class LinearAlphaBeta(nn.Linear):

    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None, alpha=1, **kwargs):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.alpha = alpha

    def forward(self, inputs):
        return linear_alpha_beta(inputs, self.weight, self.bias, self.alpha)
    
@torch.fx.wrap
def linear_alpha_beta(input, weight, bias=None, alpha=0.8):
    """

    Parameters:
    -----------
    input: torch.Tensor
        The input tensor
    weight: torch.Tensor
        The weight tensor
    bias: torch.Tensor
        The bias tensor
    epsilon: float
        Small value to stabilize the denominator
    """
    return linear_alpha_beta_fn.apply(input, weight, bias, alpha)

#-------------------------------Inspiration from Transformer Interpretability Beyond Attention Visualization---------------
def safe_divide(a, b):
    den = b.clamp(min=1e-9) + b.clamp(max=1e-9)
    den = den + den.eq(0).type(den.type()) * 1e-9
    return a / den * b.ne(0).type(b.type())

class linear_alpha_beta_fn(Function):

    @staticmethod
    def forward(ctx, inputs, weight, bias=None, alpha=0.8):
        outputs = F.linear(inputs, weight, bias)
        ctx.save_for_backward(inputs, weight, outputs)
        ctx.alpha = alpha

        return outputs

    @staticmethod
    @conservation_check_wrap
    def backward(ctx, *out_relevance):

        inputs, weight, outputs = ctx.saved_tensors
        beta = ctx.alpha - 1
        pw = torch.clamp(weight, min=0)
        nw = torch.clamp(weight, max=0)
        px = torch.clamp(inputs, min=0)
        nx = torch.clamp(inputs, max=0)

        def f(w1, w2, x1, x2):
            Z1 = F.linear(x1, w1)
            Z2 = F.linear(x2, w2)
            S1 = safe_divide(out_relevance[0], Z1 + Z2)
            S2 = safe_divide(out_relevance[0], Z1 + Z2)
            C1 = x1 * (torch.matmul(S1, w1))
            C2 = x2 * (torch.matmul(S2, w2))
            # C1 above and C1 below are the same
            # C1 = x1 * torch.autograd.grad(Z1, x1, S1)[0]
            # C2 = x2 * torch.autograd.grad(Z2, x2, S2)[0]

            return C1 + C2
        
        activator_relevances = f(pw, nw, px, nx)
        inhibitor_relevances = f(nw, pw, px, nx)

        relevance = ctx.alpha * activator_relevances - beta * inhibitor_relevances


        # relevance_norm = out_relevance[0] / _stabilize(outputs, epsilon)

        # relevance = torch.matmul(relevance_norm, weight).mul_(inputs)

        return (relevance, None, None, None)

INIT_MODULE_MAPPING[LinearAlphaBeta] = initialize_bias