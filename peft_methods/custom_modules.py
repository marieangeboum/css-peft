import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from .adapter import Adapter

class Mlp(nn.Module):
    def __init__(self, dim, mlp_hidden_dim, drop, out_dim=None):
        super().__init__()
        self.fc1 = nn.Linear(dim, mlp_hidden_dim)
        self.act = nn.GELU()
        if out_dim is None:
            out_dim = dim

        self.fc2 = nn.Linear(mlp_hidden_dim, dim)
        self.drop = nn.Dropout(drop)

       
    @property
    def unwrapped(self):
        return self

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


def resize_pos_embed(posemb, grid_old_shape, grid_new_shape, num_extra_tokens):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    posemb_tok, posemb_grid = (
        posemb[:, :num_extra_tokens],
        posemb[0, num_extra_tokens:],
    )
    if grid_old_shape is None:
        gs_old_h = int(math.sqrt(len(posemb_grid)))
        gs_old_w = gs_old_h
    else:
        gs_old_h, gs_old_w = grid_old_shape

    gs_h, gs_w = grid_new_shape
    posemb_grid = posemb_grid.reshape(1, gs_old_h, gs_old_w, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(gs_h, gs_w), mode="bilinear")
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_h * gs_w, -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.,):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        # self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        # self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        # self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, x):
        B, N, C = x.shape

        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., 
                 attn_drop=0.,drop_path=0., act_layer=nn.GELU(), 
                 norm_layer=nn.LayerNorm, config=None, layer_id=None, nb_task=None, task_id = 0):
        super().__init__()
        self.config = config
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, dropout=drop)
        
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(dim, mlp_hidden_dim, drop)
        
        self.nb_task = nb_task
        self.task_id = task_id
        if config.ffn_adapt:
            self.adaptmlp_pool = nn.ModuleList([Adapter(self.config, dropout=0.1, bottleneck=config.ffn_num,
                                    init_option=config.ffn_adapter_init_option,
                                    adapter_scalar=config.ffn_adapter_scalar,
                                    adapter_layernorm_option=config.ffn_adapter_layernorm_option,
                                    ) for i in range(self.nb_task)])

    def forward(self, x, task_id, mask = None, return_attention = False):
        y,  attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        if self.config.ffn_adapt and self.config.ffn_option == 'parallel':
            adapt_x = self.adaptmlp_pool[task_id](x, add_residual=False)
        residual = x
        x = self.drop_path(self.mlp(self.norm2(x)))
        if self.config.ffn_adapt:
            if self.config.ffn_option == 'sequential':
                x = self.adaptmlp_pool[task_id](x)
            elif self.config.ffn_option == 'parallel':
                x = x + adapt_x
            else:
                raise ValueError(self.config.ffn_adapt)
        x = residual + x
        return x
