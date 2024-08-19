from model.segmenter_adapt import *
from configs.utils import *
from datasets.utils import *
import argparse
from easydict import EasyDict

args_ = argparse.Namespace(
    batch_size=128,
    epochs=100,
    accum_iter=1,
    model='vit_base_patch16',
    weight_decay=0.0,
    lr=None,
    min_lr=0.0,
    warmup_epochs=20,
    finetune='/run/user/108646/gvfs/sftp:host=flexo/d/maboum/AdaptFormer/checkpoints/mae_pretrain_vit_b.pth',
    global_pool=False,
    data_path='/datasets01/imagenet_full_size/061417/',
    nb_classes=1000,
    output_dir='./output_dir',
    log_dir=None,
    device='cuda',
    seed=0,
    resume='',
    strategy = 'dino',
    start_epoch=0,
    eval=False,
    dist_eval=False,
    num_workers=10,
    pin_mem=True,
    world_size=1,
    local_rank=-1,
    dist_on_itp=False,
    dist_url='env://',
    dataset='cifar100',
    drop_path=0.0,
    inception=False,
    ffn_adapt=True,
    ffn_num=64,
    vpt=False,
    vpt_num=1,
    fulltune=False, 
    train_type = "finetuning",
)


config_file = "/d/maboum/JSTARS/segmentation/configs/config.yml"
config = load_config_yaml(file_path = config_file)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = config["dataset"]
data_config = dataset["flair1"]
seed = config["seed"]
directory_path = data_config["data_path"]
metadata = data_config["metadata"]
data_sequence = data_config["task_name"]
epochs = data_config['epochs']
eval_freq = data_config['eval_freq']
im_size = data_config["im_size"]
lr = data_config['learning_rate']
win_size = data_config["window_size"]
win_stride = data_config["window_stride"]
n_channels = data_config['n_channels']
n_class = data_config["n_cls"]
class_names = data_config["classnames_binary"]
eval_freq = data_config["eval_freq"]

selected_model = '_'.join([config["model_name"], "224"])
model = config["model"]
model_config = model[selected_model]
im_size = model_config["image_size"]
patch_size = model_config["patch_size"]
d_model = model_config["d_model"]
n_heads = model_config["n_heads"]
n_layers = model_config["n_layers"]
d_encoder = model_config["d_model"]

train_type = args_.train_type
lora_params = config["lora_parameters"]
lora_rank = lora_params["rank"]
lora_alpha = lora_params["rank"]

binary = data_config["binary"]

tuning_config = EasyDict(
        # AdaptFormer
        ffn_adapt=args_.ffn_adapt,
        ffn_option="parallel",
        ffn_adapter_layernorm_option="none",
        ffn_adapter_init_option="lora",
        ffn_adapter_scalar="0.1",
        ffn_num=args_.ffn_num,
        d_model=768,
        # VPT related
        vpt_on=args_.vpt,
        vpt_num=args_.vpt_num,
    )



segmodel = SegmenterAdapt(im_size, n_layers, d_model, d_encoder, 4*d_model, n_heads,n_class,
                         patch_size, selected_model, tuning_config = tuning_config, model_name=config["model_name"]).to(device)

segmodel.load_pretrained_weights()

dummy_input = torch.randn(1, 3, im_size, im_size).to(device)  # Batch de 1, 3 canaux, taille d'image im_size

# Fix all parameters
for param in segmodel.encoder.parameters():
    param.requires_grad = False
# Unfreeze the adapt_mlp layers
for name, param in segmodel.encoder.named_parameters():
    if 'adaptmlp' in name:
        param.requires_grad = True


# Test du forward
output = segmodel(dummy_input)

# Affichage des dimensions de sortie
print("Output shape:", output.shape)
