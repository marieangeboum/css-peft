import torch
import copy
import timm

import torch.nn as nn
import torch.nn.functional as F


from model.vit import VisionTransformer
from model.decoder import *

from model.vit import *
from model.utils import padding, unpadding
from timm.models.layers import trunc_normal_
from peft_methods.lora import *

class Segmenter(nn.Module):
    def __init__(
        self,
        image_size,
        n_layers, 
        d_model,
        d_encoder,
        d_ff,
        n_heads,
        n_cls,
        patch_size,
        variant,
        tuning_config, 
        model_name,
        dropout = 0.1,
        drop_path_rate=0.0,
        distilled=False,
        channels=3,
        ft_strat = None
    ):
        super().__init__()
        self.n_cls = n_cls
        self.patch_size = patch_size
        self.image_size = image_size
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_encoder = d_encoder
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.variant = variant
        
        self.tuning_config = tuning_config
        self.model_name = model_name
        self.dropout = 0.1
        self.drop_path_rate = 0.0
        self.distilled=False
        self.channels=3
        self.ft_strat = ft_strat
               
        self.encoder=VisionTransformer(
                image_size,
                patch_size,
                n_layers,
                d_model,
                d_ff,
                n_heads,
                n_cls,
                dropout=0.1,
                drop_path_rate=0.0,
                distilled=False,
                channels=3,
            )
        #else :
        #    self.encoder = 
        # self.decoder = DecoderLinear(n_cls, patch_size, d_model)
        self.decoder = MaskTransformer(n_cls, patch_size, d_encoder, n_layers,
                                       n_heads, d_model, d_ff, drop_path_rate=0.0, dropout = 0.1)

    @torch.jit.ignore
    def no_weight_decay(self):
        def append_prefix_no_weight_decay(prefix, module):
            return set(map(lambda x: prefix + x, module.no_weight_decay()))

        nwd_params = append_prefix_no_weight_decay("encoder.", self.encoder).union(
            append_prefix_no_weight_decay("decoder.", self.decoder)
        )
        return nwd_params

    def apply_biastuning(self):
        for name, param in self.encoder.named_parameters():
            if 'bias' in name:
                bias_params.append(param)
            else:
                param.requires_grad = False
                
    def apply_attntuning(self):
        for name, param in self.encoder.named_parameters():
            if 'attn' in name:
                bias_params.append(param)
            else:
                param.requires_grad = False
        return nwd_params

    def forward(self, im):
        H_ori, W_ori = im.size(2), im.size(3)
        im = padding(im, self.patch_size)
        H, W = im.size(2), im.size(3)

        x = self.encoder(im)
        # print(x)

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + self.encoder.distilled
        x = x[:, num_extra_tokens:]

        masks = self.decoder(x, (H, W))

        masks = F.interpolate(masks, size=(H, W), mode="bilinear")
        masks = unpadding(masks, (H_ori, W_ori))

        return masks
    
    def load_pretrained_weights(self, model_path=None):
        """
        Load pretrained weights into the SegmenterAdapt model.
        
        :param model_path: Path to the file containing the pre-trained weights. If None, loads default weights.
        """
        try:
            if model_path:
                # Load the weights from the specified path
                checkpoint = torch.load(model_path, map_location="cuda")
                self.load_state_dict(checkpoint, strict=False)
                print(f"Pretrained weights loaded successfully from {model_path}!")
            else:
                # Default behavior if no path is provided (e.g., loading timm model)
                timm_vision_transformer = timm.create_model('_'.join(['_'.join([self.model_name,"224"]),"dino"]), pretrained=True)
                timm_vision_transformer.head = nn.Identity()
                self.encoder.load_state_dict(timm_vision_transformer.state_dict(), strict=False)
                print("Pretrained model loaded successfully from timm!")
        except Exception as e:
            print("An error occurred while loading the pretrained model:", e)

    def load_pretrained_weights_sam(self):
        try : 
            chckpts = torch.load("./model/checkpoints/sam_vit_h_4b8939.pth")['model']
            self.encoder.load_state_dict(chckpts, False)
            print("Pretrained model loaded successfully!")
        except Exception as e :
            print("An error occurred while loading the pretrained model:", e)
            
    def fine_tuning_strategy(self):
        if self.ft_strat == "lora":
            self.apply_lora(rank=32, alpha=32, n_cls=self.n_cls)
        # elif self.ft_strat == "finetuning":
        #     self.apply_finetuning()
        # elif self.ft_strat == "biastuning":
        #     self.apply_biastuning()
        # elif self.ft_strat == "attntuning":
        #     self.apply_attntuning()
        # elif self.ft_strat == "linprob":
        #     self.apply_linprob()
        # elif self.ft_strat == "enctuning":
        #     self.apply_enctuning()
        else : 
            print("No specific fine-tuning strategy applied.")

    def apply_lora(self, rank, alpha, n_cls):
        lora_vit = LoRA_ViT_timm(vit_model=self.encoder,
                                 r=rank,
                                 alpha=alpha,
                                 num_classes=n_cls)
        self.encoder = lora_vit


    def get_attention_map_enc(self, im, layer_id):
        return self.encoder.get_attention_map(im, layer_id)

    def get_attention_map_dec(self, im, layer_id):
        x = self.encoder(im, return_features=True)

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + self.encoder.distilled
        x = x[:, num_extra_tokens:]

        return self.decoder.get_attention_map(x, layer_id)
    