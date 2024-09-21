import os
import glob
# import tqdm
import torch
import numpy as np
import seaborn as sns

from neptune.types import File # type: ignore
from tqdm import tqdm
from torchvision import transforms
import torchvision.utils as vutils
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

def attention_weights_train(model, image, target, device):

    
    return attention_weights

def train_function_recurrent(model,train_dataloader, device,optimizer, loss_fn,
                             accuracy, epoch, data_config, run, soft_target_loss_weight, ce_loss_weight,step,T = 2):

    n_channels = data_config['n_channels']
    loss_sum = 0.0
    acc_sum = 0.0
    for i, batch in tqdm(enumerate(train_dataloader), total = len(train_dataloader)) :
        image = (batch['image'][:,:n_channels,:,:]/255.).to(device)
        target = (batch['mask']).to(device)
        optimizer.zero_grad()
        with torch.no_grad():
            teacher_logits = model(image, step-1)
            soft_targets = F.softmax(teacher_logits, dim=1) /T
        student_logits = model(image, step)
        soft_prob = F.log_softmax(student_logits/T, dim=1)
        soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (T**2)
        label_loss = loss_fn(student_logits, target.squeeze(1).long())
        loss = soft_target_loss_weight * soft_targets_loss + ce_loss_weight * label_loss
        acc = accuracy(F.softmax(student_logits, dim=1).argmax(dim=1).unsqueeze(1), target.long())
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


def validation_function_recurrent(model, val_dataloader, device, loss_fn, accuracy, epoch, data_config, run,
                                  step, eval_freq=50):
    n_channels = data_config['n_channels']
    idx_list = list(range(8))
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
        output = model(image, step)
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
            domain_ids = batch['id']
            outputs = output.argmax(dim=1)
            iou_metrics = []

            # Use torch.no_grad to prevent gradients being computed during inference
            with torch.no_grad():
                # Process all images in the batch
                img_cms = [compute_conf_mat(
                    target[img].contiguous().view(-1).cpu(),
                    outputs[img].contiguous().view(-1).cpu().long(),
                    n_class
                ) for img in idx_list]
                # Compute metrics for all images
                img_metrics_list = [dl_inf.cm2metrics(img_cm.numpy())[0] for img_cm in img_cms]
                # Overlay predictions and ground truth for all images
                predictions_batch = [
                    overlay_segmentation(image[img].permute(1, 2, 0).cpu().numpy(),
                                         outputs[img].cpu().numpy(),
                                         class_labels)
                    for img in idx_list
                ]
                ground_truth_batch = [
                    overlay_segmentation(image[img].permute(1, 2, 0).cpu().numpy(),
                                         target[img, 0].cpu().numpy(),
                                         class_labels)
                    for img in idx_list
                ]
                # Stack and create grids for predictions and ground truth
                predictions_grid = vutils.make_grid(
                    [torch.tensor(prediction).permute(2, 0, 1) for prediction in predictions_batch],
                    nrow=len(idx_list) // 2)
                ground_truth_grid = vutils.make_grid([torch.tensor(gt).permute(2, 0, 1) for gt in ground_truth_batch],
                                                     nrow=len(idx_list) // 2)
                # Convert grids back to numpy for plotting
                predictions_grid_np = predictions_grid.permute(1, 2, 0).cpu().numpy()
                ground_truth_grid_np = ground_truth_grid.permute(1, 2, 0).cpu().numpy()

                # Plot the grids
                fig, axs = plt.subplots(1, 2, figsize=(15, 7.5))
                axs[0].imshow(predictions_grid_np)
                axs[0].set_title(f"Predictions - Batch {i}")
                axs[0].axis('off')

                axs[1].imshow(ground_truth_grid_np)
                axs[1].set_title(f"Ground Truth - Batch {i}")
                axs[1].axis('off')

                # Legend for class labels
                legend_patches = [mpatches.Patch(color=plt.cm.tab20(j/ len(class_labels)), label=class_labels[j])
                                  for j in class_labels]
                fig.legend(handles=legend_patches, loc='upper center', ncol=4,
                           bbox_to_anchor=(0.5, 0.05), fontsize='small')
                # Log the batch figure to Neptune
                run[f'imgs/epoch_{epoch}/batch_{i}/all_domains'].upload(fig)
                # Log IoU metrics for each image in the batch to Neptune as a DataFrame
                for idx, img in enumerate(idx_list):
                    domain_id = domain_ids[img]
                    img_metrics_per_class_df = img_metrics_list[idx]

                    # Log the whole DataFrame containing IoU metrics for this image
                    run[f'dataframes/{domain_id}_{i}_iou_{epoch}'].upload(File.as_html(img_metrics_per_class_df.round(2)))
                plt.close(fig)
            # Overall metrics and logging
            if epoch % eval_freq == 0:
                confusion_mats = sum(confusion_matrices)
                metrics_per_class_df, macro_average_metrics_df, micro_average_metrics_df = dl_inf.cm2metrics(confusion_mats)

                # Confusion matrix visualization
                fig, axs = plt.subplots(1, 2, figsize=(20, 10))
                sns.heatmap(confusion_mats/(confusion_mats.sum(axis=0) + np.finfo(float).eps), annot=True, fmt='.2f',
                            xticklabels=list(class_labels.values()), yticklabels=list(class_labels.values()), cmap="crest",
                            ax=axs[0])
                axs[0].set_title('Confusion Matrix: Precision')
                sns.heatmap(confusion_mats/(confusion_mats.sum(axis=1)[:, np.newaxis] + np.finfo(float).eps), annot=True, fmt='.2f',
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