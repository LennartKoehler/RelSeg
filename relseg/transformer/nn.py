import relseg.lxt_adjustments as lf
from bonito.nn import *

#this replaces the crfencoder, one multiplication has to be replaces
@register
class LinearCRFEncoder(Module):

    def __init__(self, insize, n_base, state_len, bias=True, scale=None, activation=None, blank_score=None, expand_blanks=True, permute=None):
        super().__init__()
        self.scale = scale
        self.n_base = n_base
        self.state_len = state_len
        self.blank_score = blank_score
        self.expand_blanks = expand_blanks
        size = (n_base + 1) * n_base**state_len if blank_score is None else n_base**(state_len + 1)
        self.linear = torch.nn.Linear(insize, size, bias=bias)
        self.activation = layers.get(activation, lambda: activation)()
        self.permute = permute

    def forward(self, x):
        if self.permute is not None:
            x = x.permute(*self.permute)
        scores = self.linear(x) #IMPORTANT this linear layer makes every single value negative. this must have something to do with the beamsearch afterwards and the training
        if self.activation is not None:
            scores = self.activation(scores)
        if self.scale is not None:
            scores = lf.mul2(scores, self.scale) # TESTVALUE
        if self.blank_score is not None and self.expand_blanks: # this adds the "empty/skip" base filled with blank scores: ACTG -> ACGTN
            T, N, C = scores.shape
            scores_view = scores.view(T, N, C // self.n_base, self.n_base)
            scores = torch.nn.functional.pad(
                scores_view,
                (1, 0, 0, 0, 0, 0, 0, 0),
                value=self.blank_score
            ) # -> (T, N, C // self.n_base, self.n_base+1) <-> adding the "N" dimension next to "ACGT" and giving it the value "blank_score"
            scores = scores.view(T, N, -1)
        return scores
