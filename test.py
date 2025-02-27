import numpy as np
import torch

def number_to_dna_ascii(sequence):
    mapping = torch.tensor([0, 65, 67, 71, 84])  # [0, ord('A'), ord('C'), ord('G'), ord('T')]
    
    # Use index_select to map the values
    return mapping[sequence.to(torch.int)]

a = torch.tensor([0.,1.,2.,0.,0.,0.,2.,3.,4.,0.,0.])
a = torch.stack([a,a,a,a])

encoded = number_to_dna_ascii(a)
def to_str(x, encoding="ascii"):
    return x[x.nonzero().squeeze(-1)].numpy().tobytes().decode(encoding)

decoded = to_str(encoded[0])
print(decoded)