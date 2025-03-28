import logging
import types
from functools import lru_cache

logger = logging.getLogger(__name__)

import torch
import torch.nn.functional as F
from torch import nn
import math

# from bonito.lrp_folder.LRP_composites import ProjSwigluMultiplication, AttentionValueMatmul
from .RMSNorm import RMSNorm
from .RoPE import RotaryEmbedding

import bonito.lrp.lxt_adjustments as lf



from bonito.nn import from_dict, register, LinearCRFEncoder, MakeContiguous, Module, Permute, Serial



def deepnorm_params(depth):
    """
    Returns the DeepNorm (https://arxiv.org/abs/2203.00555) alpha and beta parameters.
    """
    alpha = round((2*depth)**0.25, 7)
    beta = round((8*depth)**(-1/4), 7)
    return alpha, beta


@lru_cache(maxsize=2)
def sliding_window_mask(seq_len, window, device):
    band = torch.full((seq_len, seq_len), fill_value=1.0)
    band = torch.triu(band, diagonal=-window[0])
    band = band * torch.tril(band, diagonal=window[1])
    band = band.to(torch.bool).to(device)
    return band



class MultiHeadAttention(Module):
    def __init__(self, d_model, nhead, qkv_bias=False, out_bias=True, rotary_dim=None, attn_window=None):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.rotary_dim = self.head_dim if rotary_dim is None else rotary_dim
        
        self.Wqkv = torch.nn.Linear(d_model, 3 * d_model, bias=qkv_bias)

        self.out_proj = nn.Linear(d_model, d_model, bias=out_bias)

        self.rotary_emb_flash = RotaryEmbedding(self.rotary_dim, interleaved=False)

        self.attn_window = (-1, -1) if attn_window is None else tuple(attn_window)
        # self.matmul = AttentionValueMatmul()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        N, T, _ = x.shape

        qkv = self.Wqkv(x).view(N, T, 3, self.nhead, self.head_dim)
        v_val = qkv[:,:,2,:,:]
        q_val = qkv[:,:,0,:,:]
        k_val = qkv[:,:,1,:,:]

        q_val = self.rotary_emb_flash.apply_rotary_embedding_not_flash_x(q_val)
        k_val = self.rotary_emb_flash.apply_rotary_embedding_not_flash_x(k_val)


        attn_mask = sliding_window_mask(q_val.shape[1], self.attn_window, q_val.device)

        query, key, value = q_val.permute(0,2,1,3)[:,None,:,:,:], k_val.permute(0,2,1,3)[:,None,:,:,:], v_val.permute(0,2,1,3)[:,None,:,:,:]

        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1))
        attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
  

        if attn_mask is not None:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-Inf"))

        attn_weight = lf.matmul(query, key.transpose(-2, -1))
        attn_weight = lf.mul2(attn_weight, torch.tensor(scale_factor).detach())
        attn_weight = lf.add2(attn_weight, attn_bias.detach()) # IMPORTANT TODO here the relevance is lost because attn_bias is -Inf
        attn_weight = self.softmax(attn_weight)
        #attn_weight = torch.dropout(attn_weight, dropout_p, train=True) # LXT no dropout (dont need since were not training)
        
        # attn_out = self.matmul(attn_weight, value)
        attn_out = lf.matmul(attn_weight, value)
        attn_out = attn_out.permute(0, 1, 3, 2, 4)

        attn_out = attn_out.reshape(N, T, self.d_model)

        out = self.out_proj(attn_out)

        return out
    


class GatedMlp(nn.Module): # IMPORTANT simple implementation of mlp, should not be a problem for lxt, might need to think about activation functions
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        activation="sigmoid",
        bias1=True,
        bias2=True,
        multiple_of=128,
        return_residual=False,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.activation = activation
        out_features = out_features if out_features is not None else in_features
        hidden_features = (
            hidden_features if hidden_features is not None else int(8 * in_features / 3)
        )
        hidden_features = (hidden_features + multiple_of - 1) // multiple_of * multiple_of
        self.return_residual = return_residual
        self.fc1 = nn.Linear(in_features, 2 * hidden_features, bias=bias1, **factory_kwargs)
        self.activation = activation
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias2, **factory_kwargs)
        self.silu = nn.SiLU(inplace=False)
        # self.swiglu_mul = ProjSwigluMultiplication()


    def forward(self, x): #IMPORTANT maybe call apply of swiglu?
        y = self.fc1(x)

        #swiglu:
        y, gate = y.chunk(2, dim=-1)
        # y = self.swiglu_mul(y, self.silu(gate))
        y = lf.mul2(y, self.silu(gate))
        y = self.fc2(y)
        return y if not self.return_residual else (y, x)
    
    


@register
class TransformerEncoderLayer(Module):
    def __init__(self, d_model, nhead, dim_feedforward, deepnorm_alpha, deepnorm_beta, attn_window=None):
        super().__init__()
        self.kwargs = {
            "d_model": d_model,
            "nhead": nhead,
            "dim_feedforward": dim_feedforward,
            "deepnorm_alpha": deepnorm_alpha,
            "deepnorm_beta": deepnorm_beta,
            "attn_window": attn_window
        }

        self.self_attn = MultiHeadAttention(
            d_model=d_model,
            nhead=nhead,
            qkv_bias=False,
            out_bias=True,
            attn_window=attn_window
        )
        self.ff = GatedMlp(
            d_model,
            hidden_features=dim_feedforward,
            activation="swiglu",
            bias1=False,
            bias2=False,
            multiple_of=1,
        )

        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

        self.register_buffer("deepnorm_alpha", torch.tensor(deepnorm_alpha))
        self.reset_parameters()

    def reset_parameters(self):
        db = self.kwargs["deepnorm_beta"]
        d_model = self.kwargs["d_model"]
        torch.nn.init.xavier_normal_(self.ff.fc1.weight, gain=db)
        torch.nn.init.xavier_normal_(self.ff.fc2.weight, gain=db)
        torch.nn.init.xavier_normal_(self.self_attn.out_proj.weight, gain=db)
        torch.nn.init.xavier_normal_(self.self_attn.Wqkv.weight[2*d_model:], gain=db)
        torch.nn.init.xavier_normal_(self.self_attn.Wqkv.weight[:2*d_model], gain=1)

    def forward(self, x):
        x1 = self.self_attn(x)
        residuals_1 = lf.mul2(self.deepnorm_alpha.detach(), x)
        x1 = lf.add2(x1, residuals_1)
        x2 = self.norm1(x1) #IMPORTANT the xs have to be like this, also mind deepnorm_alpha*x

        x3 = self.ff(x2)
        residuals_2 = lf.mul2(self.deepnorm_alpha.detach(), x2)
        x3 = lf.add2(x3, residuals_2)
        x4 = self.norm2(x3)

        return x4

    def to_dict(self, include_weights=False):
        if include_weights:
            raise NotImplementedError
        return self.kwargs


def use_koi(self, **kwargs):
    # koi needs modified LinearCRFLayer settings
    def _expand_blanks(m):
        if isinstance(m, LinearCRFEncoder):
            m.expand_blanks = False
    self.encoder.apply(_expand_blanks)
    self.encoder = Serial([
        self.encoder,
        Permute([1, 0, 2]),
        MakeContiguous(),
    ])


def Model(config):
    model_config = {k: v for k, v in config["model"].items() if k != "package"}
    model = from_dict(model_config)
    model.config = config
    model.use_koi = types.MethodType(use_koi, model)
    return model