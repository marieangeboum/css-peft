# Third-party libraries
import cv2
import wandb
import pytorch_lightning as pl
import torch
import torchvision
# from pytorch_lightning.loggers import TensorBoardLogger
# from torch.utils.tensorboard import SummaryWriter
from PIL import Image
# from pytorch_lightning.utilities import rank_zero_warn
import numpy as np
from matplotlib import cm


class SegmentationImagesVisualisation(pl.Callback):
    """Generate images based on classifier predictions and log a batch to predefined logger.

    .. warning:: This callback supports only tensorboard right now

    """

    NB_COL: int = 2

    def __init__(self, writer,freq, *args, **kwargs):

        super().__init__(*args, **kwargs)
        #self.visu_fn = visu_fn # conversion of labels into --> ??
        self.freq = freq
        self.writer = writer

    def display_batch(self, writer, batch,freq, epoch, prefix):
        if epoch % self.freq == 0:

            img = batch['image'][:,:3,:,:].cpu()
            orig_img = batch['orig_image'][:,:3,:,:].cpu()
            target = batch['mask'].cpu()
            preds = batch['preds'].cpu() 
            np_preds =torch.from_numpy(torch.sigmoid(preds).detach().numpy()).float()
            magma = cm.get_cmap('viridis')
            np_preds_cmap =torch.from_numpy(magma(np_preds)).squeeze(1).permute(0,3,1,2)[:,:3,:,:]
            np_preds_int = torch.round(np_preds).cpu()
            if batch['mask'] is not None:
                labels = batch['mask'].cpu()
                np_labels = torch.from_numpy(labels.detach().numpy()).float()
            TP = (target.int() & np_preds_int.int()) 
            FP = ((target.int() - np_preds_int) == -1)
            FN = ((target.int()==1 )& (np_preds_int.int() == 0))
            overlay = torch.cat([FN,np_preds_int,FP], dim =1).float().clip(0,1)
            heat_map_pred = torch.mul(np_preds, overlay)
            
            # Number of grids to log depends on the batch size
            quotient, remainder = divmod(img.shape[0], self.NB_COL)
            nb_grids = quotient + int(remainder > 0)
    
            for idx in range(nb_grids):
                start = self.NB_COL * idx
                if start + self.NB_COL <= img.shape[0]:
                    end = start + self.NB_COL
                else:
                    end = start + remainder
                img_grid = torchvision.utils.make_grid(img[start:end, :3, :, :], padding=10, normalize=True)
                mask_grid = torchvision.utils.make_grid(target[start:end, :, :, :], padding=10, normalize=False)
                out_grid = torchvision.utils.make_grid(np_preds_cmap[start:end, :, :, :], padding=10, normalize=False)
                error_grid = torchvision.utils.make_grid(overlay[start:end, :, :, :], padding=10, normalize=True)
                heat_maps_grid = torchvision.utils.make_grid(heat_map_pred[start:end, :, :, :], padding=10, normalize=True)
                grids = [img_grid, mask_grid, out_grid]
                final_grid = torch.cat(grids, dim=1)
                # writer.add_image(f'Images/{prefix}_{idx}', final_grid, epoch+1)
                # wandb.log()
                # wandb.log(
                #       {"my_image_key" : wandb.Image(image, masks={
                #         "predictions" : {
                #             "mask_data" : batch['preds'],
                #             "class_labels" : class_labels
                #         },
                #         "ground_truth" : {
                #             "mask_data" : target,
                #             "class_labels" : class_labels
                #         }
                #     })})

    def on_validation_batch_end(
            self, trainer: pl.Trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ) -> None:
        """Called when the validation batch ends."""

        if trainer.current_epoch % self.freq == 0 and batch_idx == 0:
            self.display_batch(trainer, outputs['batch'], prefix='Val')
