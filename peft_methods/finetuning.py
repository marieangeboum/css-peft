def  bias_fine_tuning(model):
    bias_params = []
    for name, param in model.named_parameters():
        if 'bias' in name:
            bias_params.append(param)
        else:
            param.requires_grad = False
    return bias_params, model

def attn_fine_tuning(model):
    attn_params = []
    for name, param in model.named_parameters():
        if 'attn' in name:
            attn_params.append(param)
        else:
            param.requires_grad = False
    return attn_params, model

def topk_fine_tuning(model, k, total_blocks):
    """
    Fine-tunes only the last k blocks of a Vision Transformer model.
    
    Args:
    model (nn.Module): The Vision Transformer model.
    k (int): Number of last blocks to fine-tune.

    Returns:
    List[nn.Parameter]: List of parameters to be fine-tuned.
    """

    # Calculate the starting block index from which to start fine-tuning
    start_block = total_blocks - k

    # List to hold the parameters to be fine-tuned
    fine_tune_params = []

    # Freeze all parameters initially
    for name, param in model.named_parameters():
        if 'blocks' in name:
            block_num = int(name.split('.')[1])
            if block_num >= start_block:
                fine_tune_params.append(param)
                param.requires_grad = True
            else:
                param.requires_grad = False
        else:
            param.requires_grad = False

    # Additionally, if you want to fine-tune the final classification head
    for name, param in model.named_parameters():
        if 'head' in name or 'norm' in name:
            fine_tune_params.append(param)
            param.requires_grad = True


# BIAS_TERMS_DICT = {
#     'intermediate': 'intermediate.dense.bias',
#     'key': 'attention.self.key.bias',
#     'query': 'attention.self.query.bias',
#     'value': 'attention.self.value.bias',
#     'output': 'output.dense.bias',
#     'output_layernorm': 'output.LayerNorm.bias',
#     'attention_layernorm': 'attention.output.LayerNorm.bias',
#     'all': 'bias',
# }
# BIAS_LAYER_NAME_TO_LATEX = {
#     'attention.self.query.bias': '$\mathbf{b}_{q}^{\ell}$',
#     'attention.self.key.bias': '$\mathbf{b}_{k}^{\ell}$',
#     'attention.self.value.bias': '$\mathbf{b}_{v}^{\ell}$',
#     'attention.output.dense.bias': '$\mathbf{b}_{m_1}^{\ell}$',
#     'attention.output.LayerNorm.bias': '$\mathbf{b}_{LN_1}^{\ell}$',
#     'intermediate.dense.bias': '$\mathbf{b}_{m_2}^{\ell}$',
#     'output.dense.bias': '$\mathbf{b}_{m_3}^{\ell}$',
#     'output.LayerNorm.bias': '$\mathbf{b}_{LN_2}^{\ell}$',
# }



