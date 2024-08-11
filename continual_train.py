import os
import glob
import time
import numpy as np
import fnmatch
import random
import logging
import datetime

from rasterio.windows import Window
from argparse import ArgumentParser

import torch
import torch.nn as nn
import wandb

from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import LambdaLR
from model.segmenter import Segmenter
from model.datasets import FlairDs
from torchmetrics import Accuracy
from model.configs.utils import *
from model.datasets.utils import *
from model.datasets.memory_manager import *

def main(): 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = ArgumentParser()
    parser.add_argument("--initial_lr", type=float, default = 0.001)
    parser.add_argument("--final_lr", type=float, default = 0.0005)
    parser.add_argument("--lr_milestones", nargs=2, type=float, default=(20,80))
    parser.add_argument("--epoch_len", type=int, default=10000)
    parser.add_argument("--sup_batch_size", type=int, default=8)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--workers", default=6, type=int)
    parser.add_argument('--img_aug', type=str, default='d4_rot90_rot270_rot180_d1flip')
    parser.add_argument('--max_epochs', type=int, default=200)   
    parser.add_argument('--sequence_path', type = str, default = "sequence_{}/")
    parser.add_argument('--train_split_coef', type = float, default = 0.8)   
    parser.add_argument('--strategy', type = str, default = 'SUP_{}')
    parser.add_argument("--commit", type = str )
    parser.add_argument("--train_type", type = str, default="finetuning" )
    parser.add_argument("--replay", action="store_true", help="Enable replay")
    parser.add_argument('--config_file', type = str, 
                        default = "/d/maboum/JSTARS/SEG/configs/config.yml")
    args = parser.parse_args()

    config_file = args.config_file
    config = load_config_yaml(file_path = config_file)
    # Get current date and time
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M%S")
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
    
    dataset = config["dataset"]
    data_config = dataset["flair1"]
    seed = config["seed"]
    directory_path = data_config["data_path"]
    metadata = data_config["metadata"]
    seq_length = data_config["seq_length"]
    data_sequence = data_config["domains"]
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

    selected_model = "vit_base_patch32_224"
    model = config["model"]
    model_config = model[selected_model]
    im_size = model_config["image_size"]
    patch_size = model_config["patch_size"]
    d_model = model_config["d_model"]
    n_heads = model_config["n_heads"]
    n_layers = model_config["n_layers"]

    train_type = args.train_type
    lora_params = config["lora_parameters"]
    lora_rank = lora_params["rank"]
    lora_alpha = lora_params["rank"]
    
    binary = data_config["binary"]
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.autograd.set_detect_anomaly(True) 
    random.seed(seed)
    
    strategy = args.strategy.format(train_type)
   
    logger = logging.getLogger(__name__)
    logfile = os.path.join(config["checkpoints"],
                            f'run_{formatted_datetime}_{strategy}.log')
    logging.basicConfig(filename=logfile, 
                        filemode = 'w', force=True)
    logging.info(f"{args} \n\n")
    logging.info(f"hyperparameters for data processing : {data_config} \n\n")
    logging.info(f"model hyperparams: {model_config} \n\n")
    logging.info(f"paramètres lora : {lora_params} \n\n")

    list_of_tuples = [(item, data_sequence.index(item)) for item in data_sequence]
    if not os.path.exists(args.sequence_path.format(seed)):
        os.makedirs(os.path.join(config["checkpoints"],
                                 args.sequence_path.format(seed)))  
        
    train_imgs = []
    test_imgs = []    

    # Définition de modèles
    
    wandb.login(key="ad58a41a99168fb44b86a70954b3728fe7818df2")
    
    
    test_dataloaders = []
    train_imgs, test_imgs = [],[]
    memory = Memory(max_capacity=1000)
    for step,domain in enumerate(data_sequence):
      	
             
        segmodel = Segmenter(im_size, n_layers, d_model, 4*d_model, n_heads,n_class,
                             patch_size, selected_model, lora_rank, lora_alpha, ft_strat=train_type).to(device)
        segmodel.load_pretrained_weights()
        num_params = sum(p.numel() for p in segmodel.parameters() if p.requires_grad)
        logging.info(f"training strategy: {train_type}M \n\n")
        logging.info(f"trainable parameters: {num_params/2**20:.4f}M \n\n")        
        # if step == 0 or step ==1 or step==2 :
      	 #   continue


         
        # Définition de modèles
        model_path = os.path.join(config["checkpoints"],args.sequence_path.format(seed), 
                                 '{}_{}_{}'.format(args.strategy.format(train_type),seed, step)) 
        if step > 0 : 
            pretrained_segmodel = torch.load(os.path.join(config["checkpoints"],args.sequence_path.format(seed), 
                                      '{}_{}_{}'.format(args.strategy.format(train_type),seed, step-1)))
            segmodel.load_state_dict(pretrained_segmodel)
       
        wandb.init(project="baseline-experiments",
                    job_type = "borneinf",
                    tags = [domain], 
                    name = domain+'_'+args.strategy.format(train_type)+'_'+str(seed), 
                    config = data_config) 
          
        img = glob.glob(os.path.join(directory_path, '{}/Z*_*/img/IMG_*.tif'.format(domain)))
        random.shuffle(img)
        train_imgs += img[:int(len(img)*args.train_split_coef)]
        test_imgs  += img[int(len(img)*args.train_split_coef):]
        
        # Define Memory
        domain_memory = select_random_paths(train_imgs, 0.1, seed)
        memory.add_paths(domain_memory)
        memory_paths = memory.get_paths()
        
        domain_img = [item for item in train_imgs if  
                    fnmatch.fnmatch(item, os.path.join(directory_path, 
                    '{}/Z*_*/img/IMG_*.tif'.format(domain)))]

        domain_img_test = [item for item in test_imgs if  
                    fnmatch.fnmatch(item, os.path.join(directory_path, 
                    '{}/Z*_*/img/IMG_*.tif'.format(domain)))]
        # Train&Validation dataset
        domain_img_train = domain_img[:int(len(domain_img)*args.train_split_coef)]
        domain_img_val = domain_img[int(len(domain_img)*args.train_split_coef):]
        if args.replay and step > 0: 
            domain_img_train += memory_paths 
            random.shuffle(domain_img_train)
        train_dataloader = create_train_dataloader(domain_img_train, args, data_config, binary= binary)
        val_dataloader = create_val_dataloader(domain_img_val, args, data_config, binary= binary)
        test_dataloaders = create_test_dataloader(test_imgs, args, data_config, binary= binary)
        
        # Callbacks 
        early_stopping = EarlyStopping(patience=20, verbose=True,  delta=0.001,path=model_path)
        optimizer = SGD(segmodel.parameters(),
                        lr=args.initial_lr,
                        momentum=0.9)

        scheduler = LambdaLR(optimizer,lr_lambda= lambda_lr, verbose = True)

        if binary : 
            class_weights, cumulative_weights = domain_class_weights(metadata,
                                                             data_sequence, 
                                                            binary=binary, 
                                                            binary_label = 0)
            data_config["n_cls"] = 2
            all_weights = np.array([1./(value/100) for value in class_weights.values()])
            loss_fn = torch.nn.BCEWithLogitsLoss(weight = torch.tensor(all_weights[step]).float()).cuda()
            accuracy = Accuracy(task='binary',num_classes=n_class).cuda()
        else : 
            class_weights, cumulative_weights = domain_class_weights(metadata,
                                                             data_sequence, 
                                                            binary=binary, 
                                                            binary_label = 0)            
            weights_keys = list(class_weights[domain].keys())
            weights = list(class_weights[domain].values())
            missing_key = (n_class-1)*(n_class)//2 - sum(weights_keys)
            all_weights = np.array([1./(value/100) for value in weights])
            if len(all_weights) != n_class : 
                all_weights = np.insert(all_weights, missing_key, 0.)
                
            loss_fn = torch.nn.CrossEntropyLoss(weight = torch.tensor(all_weights).float()).cuda()
            accuracy = Accuracy(task='multiclass',num_classes=n_class).cuda()
            
        for epoch in range(1,args.max_epochs):

            time_ep = time.time() 
            
            segmodel,train_loss, train_acc = train_function(segmodel,train_dataloader, 
                                                            device,optimizer, 
                                                            loss_fn, accuracy ,scheduler, data_config)
            
            segmodel,val_metrics = validation_function(segmodel,val_dataloader, 
                                                           device,optimizer, loss_fn, 
                                                           accuracy ,scheduler, 
                                                           epoch, data_config)
            early_stopping(val_metrics['val_loss'],segmodel)
            if early_stopping.early_stop :
                break
            wandb.log({
                    "train_accuracy": train_acc["acc"], 
                    "train_loss": train_loss["loss"], 
                    "epochs" : epoch, 
                    "time" : time_ep,
                    "val_accuracy": val_metrics['val_acc'],
                    "val_loss": val_metrics['val_loss'],
                    "val_iou_building": val_metrics['val_iou_building'],
                    "val_iou_pervious": val_metrics['val_iou_pervious'],
                    "val_iou_impervious": val_metrics['val_iou_impervious'],
                    "val_iou_bare": val_metrics['val_iou_bare'],
                    "val_iou_water": val_metrics['val_iou_water'],
                    "val_iou_coniferous": val_metrics['val_iou_coniferous'],
                    "val_iou_deciduous": val_metrics['val_iou_deciduous'],
                    "val_iou_brush": val_metrics['val_iou_brush'],
                    "val_iou_vine": val_metrics['val_iou_vine'],
                    "val_iou_herbe": val_metrics['val_iou_herbe'],
                    "val_iou_agri": val_metrics['val_iou_agri'],
                    "val_iou_plowed": val_metrics['val_iou_plowed'],
                    "val_iou_other": val_metrics['val_iou_other'],
                    "mean_iou": sum([
                        val_metrics['val_iou_building'],
                        val_metrics['val_iou_pervious'],
                        val_metrics['val_iou_impervious'],
                        val_metrics['val_iou_bare'],
                        val_metrics['val_iou_water'],
                        val_metrics['val_iou_coniferous'],
                        val_metrics['val_iou_deciduous'],
                        val_metrics['val_iou_brush'],
                        val_metrics['val_iou_vine'],
                        val_metrics['val_iou_herbe'],
                        val_metrics['val_iou_agri'],
                        val_metrics['val_iou_plowed'],
                        val_metrics['val_iou_other']
                    ]) / 13  # Calculate the mean IoU
})
            time_ep = time.time() - time_ep
            
        # bestmodel = Segmenter(im_size, n_layers, d_model, 4*d_model, n_heads,n_class,
        #                       patch_size, selected_model, lora_rank, lora_alpha).to(device)
        # if  train_type == "lora" : 
        #     bestmodel.apply_lora(lora_rank, lora_alpha, n_class)
        #     bestmodel.to(device)
        # bestmodel.load_state_dict(torch.load(model_path))

        # for test_step, test_domain in enumerate(data_sequence[:step+1]) : 
            
        #     test_acc = test_function(segmodel, test_dataloaders[test_step], 
        #                             device,  accuracy,  
        #                               eval_freq, data_config, domain, )
            
        wandb.finish()

if __name__ == "__main__":
    main()   