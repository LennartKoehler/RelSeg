import numpy as np
import torch
from torch.autograd import Function
import torchviz

import torch.nn as nn
class WrapModule(nn.Module):
    """
    Base class for wrapping a rule around a module. This class is not meant to be used directly, but to be subclassed by specific rules.
    It is then used to replace the original module with the rule-wrapped module.
    """

    def __init__(self, module):
        super(WrapModule, self).__init__()
        self.module = module
class EpsilonRule(WrapModule):
    """
    Gradient X Input (Taylor Decomposition with bias or standard Epsilon-LRP rule for linear layers) according to the Equation 4-5 and 8 of the paper:
    AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers

    If one of the inputs is a constant or does not require gradients, no relevance is distributed to it.

    Parameters:
    -----------
    module: nn.Module
        The module to be wrapped
    epsilon: float
        Small value to stabilize the denominator in the input_x_gradient rule

    """

    def __init__(self, module, epsilon=1e-6):
        
        super(EpsilonRule, self).__init__(module)
        self.epsilon = epsilon

    def forward(self, *inputs):

        return epsilon_lrp_fn.apply(self.module, self.epsilon, *inputs)



class epsilon_lrp_fn(Function):
    """
    Gradient X Input (Taylor Decomposition with bias or standard Epsilon-LRP rule for linear layers) according to the Equation 4-5 and 8 of the paper:
    AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers

    If one of the inputs is a constant or does not require gradients, no relevance is distributed to it.

    Parameters:
    -----------
    fn: callable
        The function to be called with the inputs, must be differentiable in PyTorch
    epsilon: float
        Small value to stabilize the denominator
    *inputs: at least one torch.Tensor
        The input tensors to the function
    """

    @staticmethod
    def forward(ctx, fn, epsilon, *inputs):

        # create boolean mask for inputs requiring gradients
        #TODO: use ctx.needs_input_grad instead of requires_grad
        requires_grads = [True if inp.requires_grad else False for inp in inputs] # TESTVALUES
        # requires_grads = [True for inp in inputs]
        if sum(requires_grads) == 0:
            # no gradients to compute or gradient checkpointing is used
            return fn(*inputs)
        
        # detach inputs to avoid overwriting gradients if same input is used as multiple arguments (like in self-attention)
        inputs = tuple(inp.detach().requires_grad_() if inp.requires_grad else inp for inp in inputs)
        """
        if it isnt detached and the function is torch.mul(x,x) then each x gets the gradient 2y/dx = dx²/dx = 2x
        and the gradient of x is then 4x (because it accumulates 2* 2x) -> this is wrong and is fixed with inputs.detach()
        detach() fixes it because it basically says each x can be thought of as seperate (graph not connected) and therefore its like
        multiplying y*z with y=z=x but not connected in the graph and therefore the gradients dont accumulate
        """

        with torch.enable_grad():
            outputs = fn(*inputs)

        ctx.epsilon, ctx.requires_grads = epsilon, requires_grads
        # save only inputs requiring gradients
        inputs = tuple(inputs[i] for i in range(len(inputs)) if requires_grads[i])
        ctx.save_for_backward(*inputs, outputs)

        # if i dont detach, then i overwrite the outputs i save for backward and then the grad_fn is epsilon_lrp.backward, not fn.backward (the function this rule wraps)
        return outputs.detach()   
    
    @staticmethod
    def backward(ctx, *out_relevance):
        inputs, outputs = ctx.saved_tensors[:-1], ctx.saved_tensors[-1]
        relevance_norm = out_relevance[0] / outputs
        # computes vector-jacobian product
        grads = torch.autograd.grad(outputs, inputs, retain_graph=True) # if not outputs.detach() then the grad_fn of outputs is epsilon_lrp, not the function which epsilon lrp wraps
        # IMPORTANT since i detach outputs.detach() i need to retain graph here in the backward propagation, otherwise this "connection" is lost, even if i use backward(retain_graph=True)
        # return relevance at requires_grad indices else None
        relevance = iter([grads[i] for i in range(len(inputs))])
        #print("relevance: ",grads[0].mul_(inputs[0])[0, :8])
        return (None, None) + tuple(next(relevance) if req_grad else None for req_grad in ctx.requires_grads)


x = torch.tensor([2], dtype=torch.float, requires_grad=True)

w1 = torch.tensor([0.5], dtype=torch.float, requires_grad=True)
w2 = torch.tensor([0.1], dtype=torch.float, requires_grad=True)

q = x*w1
k = x*w2

class AttentionValueMatmul(nn.Module):
    def forward(self, attn, value):
        return torch.mul(attn, value)
    
eps = EpsilonRule(AttentionValueMatmul(), epsilon=1e-6)

z = eps(x,x)
t = torch.tensor([2])

# torchviz.make_dot(z).render("test2_img", format="png")

print(z)
z.backward(retain_graph=True)
print(x.grad)
