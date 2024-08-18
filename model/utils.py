import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import timm
from collections import defaultdict
from timm.models.layers import DropPath
from timm.models.layers import trunc_normal_



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


def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if "model" in state_dict:
        # For deit models
        state_dict = state_dict["model"]
    num_extra_tokens = 1 + ("dist_token" in state_dict.keys())
    patch_size = model.patch_size
    image_size = model.patch_embed.image_size
    for k, v in state_dict.items():
        if k == "pos_embed" and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(
                v,
                None,
                (image_size[0] // patch_size, image_size[1] // patch_size),
                num_extra_tokens,
            )
        out_dict[k] = v
    return out_dict


def padding(im, patch_size, fill_value=0):
    # make the image sizes divisible by patch_size
    H, W = im.size(2), im.size(3)
    pad_h, pad_w = 0, 0
    if H % patch_size > 0:
        pad_h = patch_size - (H % patch_size)
    if W % patch_size > 0:
        pad_w = patch_size - (W % patch_size)
    im_padded = im
    if pad_h > 0 or pad_w > 0:
        im_padded = F.pad(im, (0, pad_w, 0, pad_h), value=fill_value)
    return im_padded


def unpadding(y, target_size):
    H, W = target_size
    H_pad, W_pad = y.size(2), y.size(3)
    # crop predictions on extra pixels coming from padding
    extra_h = H_pad - H
    extra_w = W_pad - W
    if extra_h > 0:
        y = y[:, :, :-extra_h]
    if extra_w > 0:
        y = y[:, :, :, :-extra_w]
    return y


def resize(im, smaller_size):
    h, w = im.shape[2:]
    if h < w:
        ratio = w / h
        h_res, w_res = smaller_size, ratio * smaller_size
    else:
        ratio = h / w
        h_res, w_res = ratio * smaller_size, smaller_size
    if min(h, w) < smaller_size:
        im_res = F.interpolate(im, (int(h_res), int(w_res)), mode="bilinear")
    else:
        im_res = im
    return im_res


def sliding_window(im, flip, window_size, window_stride):
    B, C, H, W = im.shape
    ws = window_size

    windows = {"crop": [], "anchors": []}
    h_anchors = torch.arange(0, H, window_stride)
    w_anchors = torch.arange(0, W, window_stride)
    h_anchors = [h.item() for h in h_anchors if h < H - ws] + [H - ws]
    w_anchors = [w.item() for w in w_anchors if w < W - ws] + [W - ws]
    for ha in h_anchors:
        for wa in w_anchors:
            window = im[:, :, ha : ha + ws, wa : wa + ws]
            windows["crop"].append(window)
            windows["anchors"].append((ha, wa))
    windows["flip"] = flip
    windows["shape"] = (H, W)
    return windows


def merge_windows(windows, window_size, ori_shape):
    ws = window_size
    im_windows = windows["seg_maps"]
    anchors = windows["anchors"]
    C = im_windows[0].shape[0]
    H, W = windows["shape"]
    flip = windows["flip"]

    logit = torch.zeros((C, H, W), device=im_windows.device)
    count = torch.zeros((1, H, W), device=im_windows.device)
    for window, (ha, wa) in zip(im_windows, anchors):
        logit[:, ha : ha + ws, wa : wa + ws] += window
        count[:, ha : ha + ws, wa : wa + ws] += 1
    logit = logit / count
    logit = F.interpolate(
        logit.unsqueeze(0),
        ori_shape,
        mode="bilinear",
    )[0]
    if flip:
        logit = torch.flip(logit, (2,))
    result = F.softmax(logit, 0)
    return result


def inference(
    model,
    ims,
    ims_metas,
    ori_shape,
    window_size,
    window_stride,
    batch_size,
    device
):
    C = model.n_cls
    seg_map = torch.zeros((C, ori_shape[0], ori_shape[1]), device=device)
    for im, im_metas in zip(ims, ims_metas):
        im = im.to(device)
        im = resize(im, window_size)
        flip = im_metas["flip"]
        windows = sliding_window(im, flip, window_size, window_stride)
        crops = torch.stack(windows.pop("crop"))[:, 0]
        B = len(crops)
        WB = batch_size
        seg_maps = torch.zeros((B, C, window_size, window_size), device=im.device)
        with torch.no_grad():
            for i in range(0, B, WB):
                seg_maps[i : i + WB] = model.forward(crops[i : i + WB])
        windows["seg_maps"] = seg_maps
        im_seg_map = merge_windows(windows, window_size, ori_shape)
        seg_map += im_seg_map
    seg_map /= len(ims)
    return seg_map


def num_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    n_params = sum([torch.prod(torch.tensor(p.size())) for p in model_parameters])
    return n_params.item()

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout, out_dim=None):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        if out_dim is None:
            out_dim = dim
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(dropout)

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


class Attention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads
        head_dim = dim // heads
        self.scale = head_dim ** -0.5
        self.attn = None

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    @property
    def unwrapped(self):
        return self

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.heads, C // self.heads)
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
    def __init__(self, dim, heads, mlp_dim, dropout, drop_path):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout) # défini juste au-dessus
        self.mlp = FeedForward(dim, mlp_dim, dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
      
    def forward(self, x, mask=None, return_attention=False):
        y, attn = self.attn(self.norm1(x), mask)
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
    
class StateDictModifier():
    def __init__(self, model):
        self.state_dict = model.state_dict()
    @staticmethod
    def split_qkv(weights, biases):
        """
        Divise les poids et biais concaténés en poids et biais pour 'query', 'key', et 'value'.
        """
        total_dim = weights.size(0)
        assert total_dim % 3 == 0, "Les dimensions totales des poids ne sont pas divisibles par 3."
        qkv_dim = total_dim // 3
        # Diviser les poids
        q_weight = weights[:qkv_dim]
        k_weight = weights[qkv_dim:2*qkv_dim]
        v_weight = weights[2*qkv_dim:]
        # Diviser les biais
        q_bias = biases[:qkv_dim]
        k_bias = biases[qkv_dim:2*qkv_dim]
        v_bias = biases[2*qkv_dim:]
        return (q_weight, k_weight, v_weight), (q_bias, k_bias, v_bias)
    def modify_attn_weights(self):
        """
        Modifie les poids concaténés en poids individuels pour chaque bloc du state_dict.
        """
        new_state_dict = self.state_dict.copy()
        for key in list(self.state_dict.keys()):
            if key.endswith('attn.qkv.weight'):
                # Extraire les poids concaténés
                qkv_weight = self.state_dict[key]
                # Diviser les poids concaténés
                (q_weight, k_weight, v_weight), _ = self.split_qkv(
                    qkv_weight,
                    self.state_dict[key.replace('.weight', '.bias')]
                )
                # Créer les nouvelles clés pour les poids
                base_key = key.replace('.attn.qkv.weight', '.attn')
                new_state_dict[base_key + '.q_proj.weight'] = q_weight
                new_state_dict[base_key + '.k_proj.weight'] = k_weight
                new_state_dict[base_key + '.v_proj.weight'] = v_weight
                # Supprimer la clé originale
                del new_state_dict[key]
        self.state_dict = new_state_dict
    def modify_attn_biases(self):
        """
        Modifie les biais concaténés en biais individuels pour chaque bloc du state_dict.
        """
        new_state_dict = self.state_dict.copy()
        for key in list(self.state_dict.keys()):
            if key.endswith('attn.qkv.bias'):
                # Extraire les biais concaténés
                qkv_bias = self.state_dict[key]
                # Diviser les biais concaténés
                _, (q_bias, k_bias, v_bias) = self.split_qkv(
                    torch.zeros_like(qkv_bias),  # Les biais sont concaténés, pas de poids associés
                    qkv_bias
                )
                # Créer les nouvelles clés pour les biais
                base_key = key.replace('.attn.qkv.bias', '.attn')
                new_state_dict[base_key + '.q_proj.bias'] = q_bias
                new_state_dict[base_key + '.k_proj.bias'] = k_bias
                new_state_dict[base_key + '.v_proj.bias'] = v_bias
                # Supprimer la clé originale
                del new_state_dict[key]
        self.state_dict = new_state_dict
    def replace_mlp_keys(self):
        """
        Remplace les clés 'mlp.fc1.weight', 'mlp.fc1.bias', 'mlp.fc2.weight', 'mlp.fc2.bias'
        par 'fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias'.
        """
        new_state_dict = self.state_dict.copy()
        for key in list(self.state_dict.keys()):
            if 'mlp.fc' in key:
                new_key = key.replace('mlp.fc', 'fc')
                new_state_dict[new_key] = new_state_dict.pop(key)
        self.state_dict = new_state_dict
    def modify_state_dict(self):
        """
        Modifie le state_dict pour séparer les poids et biais concaténés en poids et biais individuels,
        supprimer les clés 'proj', et remplacer 'mlp.fc*' par 'fc*'.
        """
        self.modify_attn_weights()
        self.modify_attn_biases()
        self.replace_mlp_keys()
        return self.state_dict
