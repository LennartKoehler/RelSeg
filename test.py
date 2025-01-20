import torch
from flash_attn.ops.triton.layer_norm import RMSNorm
import torch.nn as nn
import torch.nn.functional as F
from lxt import functional as lf
from flash_attn.layers.rotary import RotaryEmbedding as fRotaryEmbedding

# region
#Copied from transformers.models.llama.modeling_llama.LlamaRotaryEmbedding with Llama->Mixtral
class MixtralRotaryEmbedding_mine(nn.Module):
    def __init__(self, dim, max_position_embeddings=6, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)) #theta
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype) # t = m (position in "sentence")

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = freqs.repeat_interleave(2,dim=1)
        self.freq = emb
        #emb = torch.cat((freqs, freqs), dim=1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


def transform(x):
    N, L, n_head, emb_dim = x.shape
    x = x.view(N, L, n_head,-1,2)
    x_transformed = torch.cat([-x[..., 1:2], x[..., 0:1]], dim=-1)
    return x_transformed.flatten(start_dim=-2, end_dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb_mine(q, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    #N, L, nheads, head_dim = q.shape
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)

    q_transform = transform(q)
    q_embed = lf.add2(lf.mul2(q, cos.detach()), lf.mul2(q_transform, sin.detach())) #elementwise multiplication
    return q_embed
# endregion

torch.manual_seed(1)

tensora = torch.rand((2,3,3,2,4))
tensorb = tensora.detach().clone().to("cuda")
tensorc = tensora.detach().clone().to("cuda")
print(tensora[:,:,0,:,:])
print(tensorb[:,:,0,:,:])

tensora = tensora.to("cuda")
print(torch.equal(tensora, tensorb))



rotary_flash = fRotaryEmbedding(
            4,
            interleaved=False
            ).to("cuda")
flash = rotary_flash(tensora)
not_flash = rotary_flash.apply_rotary_embedding_not_flash(tensorc)

print(flash[:,1,:,:,:],not_flash[:,1,:,:,:])



# from rotary_embedding_torch import RotaryEmbedding
# rotary_test = RotaryEmbedding(dim=4, seq_before_head_dim=True, device = "cuda").to("cuda")


# print(rotary_test.rotate_queries_or_keys(tensorb[:,:,0,:,:]))

#print(torch.equal(tensor, tensor_test))
#print(tensor_mixtral_q)

#IMPORTANT dimension doesnt quite match 