import torch
import torch.nn as nn
from lxt.functional import conservation_check_wrap


class RMSNorm(nn.Module):
    def __init__(self, d, p=-1., eps=torch.tensor(1e-8), bias=False):
        """
            Root Mean Square Layer Normalization
        :param d: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

        # if self.bias:
        #     self.offset = nn.Parameter(torch.zeros(d))
        #     self.register_parameter("offset", self.offset)

    def forward(self, x=None):
        return rms_norm_identity_custom(x, self.scale, self.eps, self.d ,self.p, self.bias)
    
@torch.fx.wrap
def rms_norm_identity_custom(x, scale, epsilon, d, p, bias):
    return rms_norm_identity_function.apply(x, scale, epsilon, d, p, bias)


class rms_norm_identity_function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, epsilon, d, p, bias):

        if p < 0. or p > 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = d
        else:
            partial_size = int(d * p)
            partial_x, _ = torch.split(x, [partial_size, d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + epsilon)


        return scale * x_normed
    

    @staticmethod
    @conservation_check_wrap
    def backward(ctx, *out_relevance):

        #print(out_relevance)
        return out_relevance + (None, None, None, None, None, None)