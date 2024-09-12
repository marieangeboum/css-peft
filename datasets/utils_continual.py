import os
import glob
# import tqdm
import torch
import numpy as np
import seaborn as sns

from neptune.types import File # type: ignore
from tqdm import tqdm
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import dl_toolbox.inference as dl_inf
from datasets import FlairDs
from rasterio.windows import Window
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, ConcatDataset
from dl_toolbox.torch_collate import CustomCollate
from dl_toolbox.callbacks import *

def overlay_segmentation(image, segmentation, class_labels, alpha = 0.5):
    overlay = image.copy()
    for class_id, class_name in class_labels.items():
        overlay[segmentation==class_id] = plt.cm.tab20(class_id/len(class_labels))[:3]
    return (1-alpha)*image + alpha*overlay


def calculate_metrics_for_image(model, image, output, target, device, binary=False):
    """
    Calculate IoU, F1, recall, and precision metrics for each class in a given image.

    Args:
    - model (torch.nn.Module): The trained model.
    - image (torch.Tensor): The input image tensor of shape (1, C, H, W).
    - target (torch.Tensor): The ground truth mask tensor of shape (1, H, W).
    - device (str): Device to perform calculations ('cuda' or 'cpu').
    - binary (bool, optional): Whether the problem is binary classification. Defaults to False.

    Returns:
    - metrics_df (pd.DataFrame): DataFrame containing IoU, F1, recall, and precision metrics for each class.
    """

    if binary:
        # For binary classification (sigmoid output)
        predicted_mask = torch.sigmoid(output).cpu().numpy().squeeze()
        binary_pred = (predicted_mask > 0.5).astype(np.uint8)
        target = target.cpu().numpy().squeeze()
        
        # Compute metrics
        cm = compute_conf_mat(target.flatten(), binary_pred.flatten(), num_classes=2)
        metrics_per_class_df, _, _ = dl_inf.cm2metrics(cm)
        
    else:
        # For multi-class classification (softmax output)
        predicted_mask = F.softmax(output, dim=1).argmax(dim=1).cpu().numpy().squeeze()
        target = target.cpu().numpy().squeeze()
        
        # Compute metrics
        cm = compute_conf_mat(target.flatten(), predicted_mask.flatten(), num_classes=output.shape[1])
        metrics_per_class_df, _, _ = dl_inf.cm2metrics(cm)
    
    return metrics_per_class_df


def train_function(model,train_dataloader, device,optimizer, loss_fn, accuracy, epoch, data_config, run ):
    n_channels = data_config['n_channels']

    class_labels = data_config["classnames"]
    n_class = data_config["n_cls"]
    loss_sum = 0.0
    acc_sum = 0.0
    for i, batch in tqdm(enumerate(train_dataloader), total = len(train_dataloader)) :
        image = (batch['image'][:,:n_channels,:,:]/255.).to(device)
        target = (batch['mask']).to(device)

        optimizer.zero_grad()
        logits = model(image)
        loss = loss_fn(F.softmax(logits, dim=1),target.squeeze(1).long())
        acc = accuracy(F.softmax(logits, dim=1).argmax(dim=1).unsqueeze(1), target.long())
        loss_sum += loss.item()
        acc_sum += acc
        loss.backward()
        optimizer.step() 
        loss_sum += loss.item()
    train_loss = loss_sum / len(train_dataloader)
    train_acc = acc_sum/len(train_dataloader)
    run["train/accuracy"].append(train_acc, step=epoch)
    run["train/loss"].append(train_loss, step=epoch)

    return model, train_acc, train_loss


def validation_function(model, val_dataloader, device, loss_fn, accuracy, epoch, data_config, run, eval_freq=50):
    n_channels = data_config['n_channels']
    interpolation = data_config["interpolation"]
    img_logs = data_config["img_logs"]
    idx_list = [0, -1]
    class_labels = data_config["classnames"]
    n_class = data_config["n_cls"]
    # Initialize accumulators
    loss_sum, acc_sum = 0.0, 0.0
    iou_metrics = torch.zeros(n_class)
    confusion_matrices = []
    for i, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
        # Preprocess input and target
        image = (batch['image'][:, :n_channels, :, :] / 255.).to(device)
        target = batch['mask'].to(device)
        
        # Model forward pass
        output = model(image)
        softmax_output = F.softmax(output, dim=1)

        # Compute loss and accuracy
        loss = loss_fn(softmax_output, target.squeeze(1).long())
        acc = accuracy(softmax_output.argmax(dim=1).unsqueeze(1), target)

        # Update accumulators
        loss_sum += loss.item()
        acc_sum += acc

        # Compute confusion matrix
        cm = compute_conf_mat(target.contiguous().view(-1).cpu(),
                              output.argmax(dim=1).contiguous().view(-1).cpu().long(), n_class)
        confusion_matrices.append(cm.numpy())

        # Compute IoU for each class
        metrics_per_class_df, _, _ = dl_inf.cm2metrics(cm.numpy())
        iou_metrics += torch.tensor(metrics_per_class_df.IoU.values)

        # Evaluate and log images at specific intervals
        if epoch % 10 == 0 and i % eval_freq == 0:
            for img in idx_list:
                domain_id = batch['id'][img]
                # Compute metrics for specific images
                img_cm = compute_conf_mat(
                    target[img].contiguous().view(-1).cpu(),
                    output.argmax(dim=1)[img].contiguous().view(-1).cpu().long(),
                    n_class
                )
                img_metrics_per_class_df, _, _ = dl_inf.cm2metrics(img_cm.numpy())
                # Plot predictions and ground truth
                predictions = overlay_segmentation(image[img].permute(1, 2, 0).cpu().numpy(),
                                                   output.argmax(dim=1)[img].cpu().numpy(),
                                                   class_labels)
                ground_truth = overlay_segmentation(image[img].permute(1, 2, 0).cpu().numpy(),
                                                    target[img, 0].cpu().numpy(),
                                                    class_labels)
                fig, axs = plt.subplots(1, 2, figsize=(15, 7.5))
                axs[0].imshow(predictions)
                axs[0].set_title("Predictions")
                axs[0].axis('off')
                axs[1].imshow(ground_truth)
                axs[1].set_title("Ground Truth")
                axs[1].axis('off')

                # Legend
                legend_patches = [mpatches.Patch(color=plt.cm.tab20(j / len(class_labels)), label=class_labels[j])
                                  for j in class_labels]
                fig.legend(handles=legend_patches, loc='upper center', ncol=4,
                           bbox_to_anchor=(0.5, 0.11), fontsize='small')

                # Log figure to Neptune
                run[f'imgs/epoch_{epoch}/batch_{i}/domain_{domain_id}'].upload(fig)

                # Log IoU metrics for this image to Neptune
                for cls_idx, cls_name in class_labels.items():
                    run[f'metrics/{domain_id}_{i}_{cls_name}_iou'].append(img_metrics_per_class_df.IoU.loc[cls_idx].round(2))

        # Overall metrics and logging
            if epoch % eval_freq == 0:
                confusion_mats = sum(confusion_matrices)
                metrics_per_class_df, macro_average_metrics_df, micro_average_metrics_df = dl_inf.cm2metrics(confusion_mats)

                # Confusion matrix visualization
                fig, axs = plt.subplots(1, 2, figsize=(20, 10))
                sns.heatmap(confusion_mats /  (confusion_mats.sum(axis=0) + np.finfo(float).eps), annot=True, fmt='.2f',
                            xticklabels=list(class_labels.values()), yticklabels=list(class_labels.values()), cmap="crest",
                            ax=axs[0])
                axs[0].set_title('Confusion Matrix: Precision')
                sns.heatmap(confusion_mats / (confusion_mats.sum(axis=1)[:, np.newaxis] + np.finfo(float).eps), annot=True, fmt='.2f',
                            xticklabels=list(class_labels.values()), yticklabels=list(class_labels.values()), cmap="cubehelix",
                            ax=axs[1])
                axs[1].set_title('Confusion Matrix: Recall')
                # Log confusion matrix to Neptune
                run[f'metrics/epoch_{epoch}/confusion_matrix'].upload(fig)
                # Log overall metrics to Neptune
                run[f'metrics/epoch_{epoch}/metrics_per_class'].upload(File.as_html(metrics_per_class_df.round(2)))
                run[f'metrics/epoch_{epoch}/macro_average_metrics'].upload(File.as_html(macro_average_metrics_df.round(2)))
                run[f'metrics/epoch_{epoch}/micro_average_metrics'].upload(File.as_html(micro_average_metrics_df.round(2)))
    val_loss = loss_sum / len(val_dataloader)
    val_acc = acc_sum.item() / len(val_dataloader)
    val_iou = torch.mean(iou_metrics)/len(val_dataloader)
    # Log final metrics to Neptune
    run["val/accuracy"].append(val_acc, step=epoch)
    run["val/loss"].append(val_loss, step=epoch)
    run["val/iou"].append(val_iou, step=epoch)
    return model, val_loss, val_acc, val_iou


def test_function(model,test_dataloader, device, accuracy , eval_freq, data_config, domain, step):
    acc_sum = 0.0
    n_channels = data_config['n_channels']
    binary =  data_config["binary"]
    interpolation = data_config["interpolation"]
    norm_means = data_config["norm_task"]["norm_means"]
    norm_stds = data_config["norm_task"]["norm_stds"]
    norm_transforms = transforms.Compose([transforms.Normalize(norm_means[:n_channels], norm_stds[:n_channels])])
    if binary: 
        class_labels = data_config["classnames_binary"]
        n_class = 2
    else :
        class_labels = data_config["classnames"]
        n_class = data_config["n_cls"]
    confusion_matrices = []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_dataloader), total = len(test_dataloader)):
            image = (batch['image'][:,:n_channels,:,:]/255.).to(device)
            if interpolation :
                target = (batch['mask']).to(device)
            target_view = target
            output = model(image)
            # target = F.one_hot(target.long()[:,0,:,:],n_class).permute(0, 3, 1, 2).to(device)
            if binary :
                target = F.one_hot(target[:,0,:,:].long(),n_class).permute(0, 3, 1, 2).to(device)
                acc = accuracy(torch.sigmoid(output).contiguous().view(-1),
                                target.contiguous().view(-1))
                cm = compute_conf_mat(
                    target.contiguous().view(-1).cpu(),
                    (torch.sigmoid(output)>.5).contiguous().view(-1).cpu(), n_class)
            else :
                acc = accuracy(F.softmax(output, dim=1).argmax(dim=1).unsqueeze(1), target.long())
                cm = compute_conf_mat(
                    target.contiguous().view(-1).cpu(),
                    output.argmax(dim = 1).contiguous().view(-1).cpu(), n_class)
            acc_sum += acc
            confusion_matrices.append(cm.numpy())
            # wandb_image_test.append(
            #     wandb.Image(image[0,:,:,:].permute(1, 2, 0).cpu().numpy(), 
            #     masks={"prediction" :
            #     {"mask_data" : F.softmax(output, dim=1).argmax(dim=1)[0,:,:].cpu().numpy(), "class_labels" : class_labels},
            #     "ground truth" : 
            #     {"mask_data" : target_view[0,0,:,:].cpu().numpy(), "class_labels" : class_labels}}, 
            #     caption= "batch_{}_domain_{}".format(i, batch['id'][0])))
            # if i % 200 == 0 :
            #     wandb.log({f"(Inf) Predictions {step} {domain}": wandb_image_test})
        confusion_mats = sum(confusion_matrices)
        metrics_per_class_df, macro_average_metrics_df, micro_average_metrics_df = dl_inf.cm2metrics(confusion_mats)
        fig = plt.figure(figsize=(20, 10))
        ax1 = fig.add_subplot(121)
        sns.heatmap(confusion_mats/confusion_mats.sum(axis = 0), annot=True, fmt='.2f', 
                    xticklabels=list(class_labels.values()), yticklabels=list(class_labels.values()), 
                    cmap = "crest")
        ax1.set_title('Confusion Matrix : Precision')
        ax2 = fig.add_subplot(122)
        sns.heatmap(confusion_mats/confusion_mats.sum(axis = 1), annot=True, fmt='.2f', 
                    xticklabels=list(class_labels.values()), yticklabels=list(class_labels.values()), 
                    cmap = sns.cubehelix_palette(as_cmap=True))
        ax2.set_title('Confusion Matrix : Recall')
        # wandb.log({f"Confusion Matrix (Inference) {step} {domain}":wandb.Image(fig)})
        # wandb.log({f'Metrics Class (Inference) {step} {domain}': wandb.Table(dataframe= metrics_per_class_df)})
        # wandb.log({f'Macro Average Class (Inference) {step} {domain}': wandb.Table(dataframe= macro_average_metrics_df)})
        # wandb.log({f'Micro Average Class (Inference) {step} {domain}': wandb.Table(dataframe= micro_average_metrics_df)})
        # test_acc = {'acc': acc_sum/ len(test_dataloader)}
        # wandb.log(test_acc)
    return test_acc
