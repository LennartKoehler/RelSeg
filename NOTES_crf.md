the beamsearch is probably used on the crf to find the best combination of sequences, the nn outputs a value for the probability (not really probability) of each datapoint (downsampled a bit) being a specific kmer. Through this matrix the path with the highest total probability is chosen through beam search, as i understand it

the result of the beam search is a sequence of int, there are only five relevant ints which are the int for the ascii of ACGTN



use_koi=True
tensor([[[-2.1074, -3.0645, -4.4688,  ..., -1.2803, -2.6914, -0.8906],
         [-1.6973, -2.9336, -4.2070,  ...,  0.1262, -3.3398, -0.2324],
         [-2.9883, -3.9082, -4.3242,  ..., -2.6914, -3.3086, -1.9102],
         ...,
         [-0.9062, -1.5479, -2.2031,  ..., -1.7783, -1.6895, -2.0234],
         [-1.3076, -1.7227, -1.4922,  ..., -0.6426, -1.8838, -0.9482],
         [-1.4111, -1.9346, -1.4990,  ..., -2.2090, -2.2324, -1.4980]]],
       device='cuda:0', dtype=torch.float16, grad_fn=<PermuteBackward0>)

use_koi=False:
tensor([[[ 2.0000, -2.1074, -3.0645,  ..., -1.2803, -2.6914, -0.8906],
         [ 2.0000, -1.6973, -2.9336,  ...,  0.1262, -3.3398, -0.2324],
         [ 2.0000, -2.9883, -3.9082,  ..., -2.6914, -3.3086, -1.9102],
         ...,
         [ 2.0000, -0.9062, -1.5479,  ..., -1.7783, -1.6895, -2.0234],
         [ 2.0000, -1.3076, -1.7227,  ..., -0.6426, -1.8838, -0.9482],
         [ 2.0000, -1.4111, -1.9346,  ..., -2.2090, -2.2324, -1.4980]]],
       device='cuda:0', dtype=torch.float16, grad_fn=<PermuteBackward0>)