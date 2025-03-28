import lxt.functional as lf
import torch
import torch.fx
from torch.autograd import Function
import torch.nn.functional as F

EPSILON = 1e-6

def add2(input_a, input_b):
    return add2_tensors_fn.apply(input_a, input_b, False, EPSILON)

def matmul(input_a, input_b):
    return lf.matmul(input_a, input_b, epsilon=EPSILON)

mul2 = lf.mul2


class add2_tensors_fn(Function):
    """
    Standard Epsilon-LRP rule for elementwise addition (along all dimensions) of two tensors according to the Equation 8 of the paper
    'AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers'

    Parameters:
    -----------
    input_a: torch.Tensor
        The first input tensor
    input_b: torch.Tensor
        The second input tensor
    inplace: bool
        Whether to perform the operation in place during the backward pass, will overwrite the relevance at the output
    epsilon: float
        Small value to stabilize the denominator
    """
    
    @staticmethod
    def forward(ctx, input_a, input_b, inplace=False, epsilon=1e-6):
    
        outputs = input_a + input_b
        if any([inp.requires_grad for inp in (input_a, input_b)]):
            ctx.save_for_backward(input_a, input_b)
            ctx.epsilon, ctx.inplace = epsilon, inplace

        return outputs

    @staticmethod
    @lf.conservation_check_wrap
    def backward(ctx, *out_relevance):

        #TODO: replace for conservation check with requires grad stuff

        input_a, input_b = ctx.saved_tensors


        if ctx.inplace:
            relevance_norm = out_relevance[0].div_(lf._stabilize(input_a + input_b, epsilon=ctx.epsilon, inplace=True))

            relevance_a = relevance_norm * input_a
            relevance_b = relevance_norm.mul_(input_b)

        else:
            relevance_norm = out_relevance[0] / lf._stabilize(input_a + input_b, epsilon=ctx.epsilon, inplace=True)

            relevance_a = relevance_norm * input_a
            relevance_b = relevance_norm * input_b

        relevance_a = torch.where(torch.isnan(relevance_a), torch.tensor(0).to(relevance_a), relevance_a) #stabilize attention mask, this is what is changed from orig. lxt
        relevance_b = torch.where(torch.isnan(relevance_b), torch.tensor(0).to(relevance_b), relevance_b) 

        return (relevance_a, relevance_b, None, None)