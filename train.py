import time
import neptune # type: ignore
import datetime
from torchmetrics import Accuracy # type: ignore
from configs.utils import *
from datasets.utils import *
from easydict import EasyDict # type: ignore
from argparse import ArgumentParser
from torch.optim import SGD, Adam # type: ignore
from torch.optim.lr_scheduler import LambdaLR # type: ignore
from model.segmenter_adapt import SegmenterAdapt
from dl_toolbox.callbacks import EarlyStopping # type: ignore


def save_checkpoint(state, filename="checkpoint.pth.tar"):
    """Save the training state (model, optimizer, etc.)."""
    torch.save(state, filename)

def load_checkpoint(filename="checkpoint.pth.tar"):
    """Load the training state (model, optimizer, etc.)."""
    if os.path.isfile(filename):
        print(f"=> Loading checkpoint '{filename}'")
        checkpoint = torch.load(filename)
        return checkpoint
    else:
        print(f"=> No checkpoint found at '{filename}'")
        return None

def main(): 
    api_token = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwN2IzOGYxMC0xYTg5LTQxMGEtYjE3Yy1iNDVkZDM1MmEzYzIifQ=="
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = ArgumentParser()
    parser.add_argument("--initial_lr", type=float, default = 0.001)
    parser.add_argument("--final_lr", type=float, default = 0.005)
    parser.add_argument("--lr_milestones", nargs=2, type=float, default=(20,80))
    parser.add_argument("--epoch_len", type=int, default=10000)
    parser.add_argument("--sup_batch_size", type=int, default=8)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--workers", default=6, type=int)
    parser.add_argument('--img_aug', type=str, default='d4_rot90_rot270_rot180_d1flip')
    parser.add_argument('--max_epochs', type=int, default=150)   
    parser.add_argument('--sequence_path', type = str, default = "")
    parser.add_argument('--train_split_coef', type = float, default = 0.85)
    parser.add_argument('--strategy', type = str, default = 'continual_{}')
    parser.add_argument("--commit", type = str )
    parser.add_argument("--train_type", type = str, default="adaptmlp")
    parser.add_argument("--replay", action="store_true", help="Enable replay")
    parser.add_argument('--config_file', type = str, 
                        default = "/d/maboum/css-peft/configs/config.yml")
    parser.add_argument('--checkpoint_file', type=str, default="/scratcht/FLAIR_1/experiments/checkpoints/vit-adapter/checkpoint.pth",
                        help="Checkpoint file to resume training")
    parser.add_argument('--ffn_adapt', default=True, action='store_true', help='whether activate AdaptFormer')
    parser.add_argument('--ffn_num', default=64, type=int, help='bottleneck middle dimension')
    parser.add_argument('--vpt', default=False, action='store_true', help='whether activate VPT')
    parser.add_argument('--vpt_num', default=1, type=int, help='number of VPT prompts')
    parser.add_argument('--fulltune', default=False, action='store_true', help='full finetune model')
    args = parser.parse_args()

    config_file = args.config_file
    config = load_config_yaml(file_path = config_file)
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M%S")

    seed = config["seed"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.autograd.set_detect_anomaly(True) 
    random.seed(seed)

    # Learning rate
    def lambda_lr(epoch):
        m = epoch / args.max_epochs
        if m < args.lr_milestones[0]:
            return 1
        elif m < args.lr_milestones[1]:
            return 1 + ((m - args.lr_milestones[0]) / (
                        args.lr_milestones[1] - args.lr_milestones[0])) * (
                               args.final_lr / args.initial_lr - 1)
        else:
            return args.final_lr / args.initial_lr
        
    train_type = args.train_type
    strategy = args.strategy.format(train_type)
    dataset = config["dataset"]
    data_config = dataset["flair1"]
    
    directory_path = data_config["data_path"]
    metadata = data_config["metadata"]
    data_sequence = data_config["task_name"]
    n_class = data_config["n_cls"]
    selected_model = "vit_base_patch8_224"
    model_type = config["model"]
    model_config = model_type[selected_model]
    im_size = model_config["image_size"]
    
    patch_size = model_config["patch_size"]
    d_model = model_config["d_model"]
    n_heads = model_config["n_heads"]
    n_layers = model_config["n_layers"]
    d_encoder = model_config["d_model"]
    binary = data_config["binary"]

    tuning_config = EasyDict(
        # AdaptFormer
        ffn_adapt=args.ffn_adapt,
        ffn_option="parallel",
        ffn_adapter_layernorm_option="none",
        ffn_adapter_init_option="lora",
        ffn_adapter_scalar="0.1",
        ffn_num=args.ffn_num,
        d_model=d_model,
        # VPT related
        vpt_on=args.vpt,
        vpt_num=args.vpt_num,
        nb_task = len(data_sequence[:5]),
        tasks = data_sequence[:5]) 
    
    
    for step,domain in enumerate(tuning_config.tasks):
        train_imgs, test_imgs = [],[]
        test_dataloaders = []
        run = neptune.init_run(
                project="continual-semantic-segmentation/peft-methods",
                api_token=api_token,
                name=f"AdaptFormerSeg{step}",
                description="First run for Adapters project",
                tags=["adaptmlp", "test", "segmenter", "vit-base", "patch8"])

        img = glob.glob(os.path.join(directory_path, '{}/Z*_*/img/IMG_*.tif'.format(domain)))
        random.shuffle(img)
        train_imgs += img[:int(len(img)*args.train_split_coef)]
        test_imgs += img[int(len(img)*args.train_split_coef):]
        random.shuffle(train_imgs)

        # Train&Validation Data
        domain_img_train = train_imgs[:int(len(train_imgs)*args.train_split_coef)]
        domain_img_val = train_imgs[int(len(train_imgs)*args.train_split_coef):]

        train_loader = create_train_dataloader(domain_img_train, args, data_config, binary= binary)
        val_loader = create_val_dataloader(domain_img_val, args, data_config, binary= binary)

        test_dataloader = create_test_dataloader(test_imgs, args, data_config, binary= binary)
        test_dataloaders.append(test_dataloader)

        # Model Definition
        segmentation_model_path = os.path.join(config["checkpoints"],args.sequence_path.format(seed),
                                 '{}_{}'.format(args.strategy.format(train_type),seed))
        segmentation_model = SegmenterAdapt(im_size, n_layers, d_model, d_encoder, 4 * d_model, n_heads, n_class,
                                            patch_size, selected_model, tuning_config=tuning_config,
                                            model_name=config["model_name"], id = step).to(device)
        segmentation_model.load_pretrained_weights()
        for param in segmentation_model.encoder.parameters():
            param.requires_grad = False
        # Unfreeze the adapt_mlp layers
        for name, param in segmentation_model.encoder.named_parameters():
            if 'adaptmlp' in name:
                param.requires_grad = True
        num_params = sum(p.numel() for p in segmentation_model.parameters() if p.requires_grad)
        print(f"training strategy: {train_type}\n\n")
        print(f"trainable parameters: {num_params/2**20:.4f}M \n\n")
        
        #Class Weights
        class_weights, cumulative_weights = domain_class_weights(metadata,data_sequence, 
                                                                 binary=binary,binary_label = 0)
        weights_keys = list(class_weights[domain].keys())
        weights = list(class_weights[domain].values())
        missing_key = (n_class-1)*(n_class)//2 - sum(weights_keys)
        all_weights = np.array([1./(value/100) for value in weights])
        if len(all_weights) != n_class :
            all_weights = np.insert(all_weights, missing_key, 0.)

        # Callbacks
        early_stopping = EarlyStopping(patience=20, verbose=True, 
                                       delta=0.001,path=segmentation_model_path)
        optimizer = SGD(segmentation_model.parameters(),
                        lr=args.initial_lr,
                        momentum=0.9)
        scheduler = LambdaLR(optimizer,lr_lambda= lambda_lr, verbose = True)
        loss_fn = torch.nn.CrossEntropyLoss(weight = torch.tensor(all_weights).float()).cuda()
        #loss_fn = torch.nn.CrossEntropyLoss().cuda()
        accuracy = Accuracy(task='multiclass',num_classes=n_class).cuda()

        # Check if a checkpoint exists and load it
        checkpoint = load_checkpoint(args.checkpoint_file)

        if checkpoint:
            start_epoch = checkpoint['epoch']
            segmentation_model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            early_stopping.load_state(checkpoint['early_stopping_state_dict'])
        else:
            start_epoch = 1

        for epoch in range(start_epoch,args.max_epochs):
            time_ep = time.time()
            segmentation_model, train_acc, train_loss = train_function(segmentation_model,train_loader, 
                                                device,optimizer, loss_fn,
                                                accuracy,epoch, data_config, run)
            print(train_acc, train_loss)
            scheduler.step()
            segmentation_model, val_loss, val_acc, val_iou = validation_function(segmentation_model,val_loader, 
                                                               device,loss_fn,
                                                               accuracy, epoch, data_config, run)
            
            print(val_acc, val_loss)
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': segmentation_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'early_stopping_state_dict': early_stopping.save_state(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'val_iou': val_iou
            }, filename=args.checkpoint_file)

            early_stopping(val_loss,segmentation_model)
            if early_stopping.early_stop:
                break
            time_ep = time.time() - time_ep
        segmentation_model.increment()
        # segmentation_model,val_metrics = test_function(segmentation_model,test_dataloader, device, accuracy,
        #                                      eval_freq, data_config, "SUP", "0")
        run.stop()
if __name__ == "__main__":
    main()