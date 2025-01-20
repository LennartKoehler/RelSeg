import logging
import types
from functools import lru_cache

logger = logging.getLogger(__name__)

import torch
import torch.nn.functional as F
from torch import nn
import math

from bonito.lxt_folder.RMSNorm import RMSNorm

import lxt.modules as lm
import lxt.functional as lf
try:
    from flash_attn.modules.mlp import GatedMlp as faGatedMlp
    from flash_attn.layers.rotary import RotaryEmbedding
    from flash_attn import flash_attn_qkvpacked_func
    from flash_attn.ops.triton.layer_norm import RMSNorm as faRMSNorm
except ImportError:
    logger.warning(
        "please install flash-attn to use the transformer module: "
        "`pip install flash-attn --no-build-isolation`"
    )

from bonito.nn import from_dict, register, LinearCRFEncoder, MakeContiguous, Module, Permute, Serial

def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-Inf")) # TESTVALUE was "-Inf"
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        attn_bias.masked_fill_(attn_mask.logical_not(), float("-Inf")) # TESTVALUE was "-Inf"
 

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1)
    attn_weight = attn_weight * torch.tensor(scale_factor, requires_grad=False)
    attn_weight = attn_weight + attn_bias.detach() #IMPORTANT PROBLEM HERE, attn_bias is -Inf
    attn_weight = torch.softmax(attn_weight, dim=-1)
    #attn_weight = torch.dropout(attn_weight, dropout_p, train=True) # LXT no dropout (dont need since were not training)
    return attn_weight @ value

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

        # self.q_weight = self.Wqkv.weight[:d_model, :].clone().to("cuda")
        # self.k_weight = self.Wqkv.weight[d_model:2*d_model, :].clone().to("cuda")
        # self.v_weight = self.Wqkv.weight[2*d_model:, :].clone().to("cuda") #IMPORTANT this is the only way the state_dict loading still works

        # if qkv_bias:
        #     self.q_bias = self.Wqkv.bias[:d_model].clone().to("cuda")
        #     self.k_bias = self.Wqkv.bias[d_model:2*d_model].clone().to("cuda")
        #     self.v_bias = self.Wqkv.bias[2*d_model:].clone().to("cuda")

        self.out_proj = nn.Linear(d_model, d_model, bias=out_bias)

        self.rotary_emb_flash = RotaryEmbedding(self.rotary_dim, interleaved=False)

        self.attn_window = (-1, -1) if attn_window is None else tuple(attn_window)

    def attn_func(self, q, k, v):
        # if torch.cuda.get_device_capability(qkv.device)[0] >= 8 and (torch.is_autocast_enabled() or qkv.dtype == torch.half):
        #     attn_output = flash_attn_qkvpacked_func(qkv, window_size=self.attn_window)
        # else:

        mask = sliding_window_mask(q.shape[1], self.attn_window, q.device)
        attn_output = scaled_dot_product_attention(q.permute(0,2,1,3)[:,None,:,:,:], k.permute(0,2,1,3)[:,None,:,:,:], v.permute(0,2,1,3)[:,None,:,:,:], attn_mask=mask)
        attn_output = attn_output.permute(0, 1, 3, 2, 4)
        return attn_output

    def forward(self, x):
        N, T, _ = x.shape
        qkv = self.Wqkv(x).view(N, T, 3, self.nhead, self.head_dim)


        # self.q_weight.requires_grad_()
        # self.k_weight.requires_grad_()
        # self.v_weight.requires_grad_()
        # q_val = torch.matmul(x, self.q_weight.to(x.dtype).T).view(N,T,self.nhead, self.head_dim)        
        # k_val = torch.matmul(x, self.k_weight.to(x.dtype).T).view(N,T,self.nhead, self.head_dim)
        # v_val = torch.matmul(x, self.v_weight.to(x.dtype).T).view(N,T,self.nhead, self.head_dim)
        q_val = qkv[:,:,0,:,:]
        k_val = qkv[:,:,1,:,:]
        v_val = qkv[:,:,2,:,:]
        q_val = self.rotary_emb_flash.apply_rotary_embedding_not_flash_x(q_val)
        k_val = self.rotary_emb_flash.apply_rotary_embedding_not_flash_x(k_val)
        # qkv = self.rotary_emb_flash(qkv)


        attn_output = self.attn_func(q_val, k_val, v_val).reshape(N, T, self.d_model)
        #attn_output = q_val.reshape(N,T, self.d_model)

        out = self.out_proj(attn_output)
        #out = attn_output
        return out
    



    
def swiglu(gate, y):
    # temp = lf.mul2(gate, y)
    temp = gate * y
    y = temp * 1/(torch.add(torch.tensor(1, requires_grad=False), torch.exp(-gate)))
    #y = lf.mul2(temp, 1/(lf.add2(torch.tensor(1).detach(), torch.exp(-gate))))
    return y


    




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

    def forward(self, x): #IMPORTANT maybe call apply of swiglu?
        y = self.fc1(x)
        if self.activation == "sigmoid":  # Special case for GLU
            y = F.glu(y, dim=-1)
        elif self.activation == "swiglu":  # Special case for SwiGLU
            y, gate = y.chunk(2, dim=-1)
            y = swiglu(gate, y) # IMPORTANT these if clauses only exist because there are special implementations for sigmoid and silu
                                # running with sigmoid and not replacing it through glu has the exact same output but would just go through the else statement
        else:
            y, gate = y.chunk(2, dim=-1)
            y = y * self.activation(gate)
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
        residuals_1 = self.deepnorm_alpha.detach()*x
        x1 = x1 + residuals_1
        x2 = self.norm1(x1) #IMPORTANT the xs have to be like this, also mind deepnorm_alpha*x

        x3 = self.ff(x2)
        residuals_2 = self.deepnorm_alpha.detach()*x2
        x3 = x3 + residuals_2
        x4 = self.norm2(x3) #IMPORTANT problem here because i tihnk the xs might be changed inplace, atleast these "skip" connections are problematical


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