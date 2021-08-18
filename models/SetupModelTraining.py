from pathlib import Path

import torch

from pytorch_lightning import LightningModule
from torchvision.ops import box_iou
import torch.nn as nn

class SetupModelTraining(LightningModule):
    def __init__(self, model, input_shape=None, lr=1e-4, pretrained=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.input_shape = input_shape
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.opt = optimizer
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 20, 40], gamma=0.5)
        # return [optimizer], [scheduler]
        return optimizer

    def weighted_sums(self, larger_bbx, smaller_bbx, l):
        loss = torch.sum(torch.abs(larger_bbx[l:, 0]))
        loss += torch.sum(torch.abs(larger_bbx[l:, 1]) ** 2 + torch.abs(larger_bbx[l:, 2]) ** 2)
        loss += torch.sum(torch.abs(larger_bbx[l:, 1] ** 0.5) + torch.abs(larger_bbx[l:, 2] ** 0.5))
        if larger_bbx.size(0) > 0 and smaller_bbx.size(0) > 0:
            loss += torch.sum(torch.abs(larger_bbx[:l, 0] - smaller_bbx[:, 0]))
            loss += torch.sum(
                torch.abs(larger_bbx[:l, 1] - smaller_bbx[:, 1]) ** 2 + torch.abs(larger_bbx[:l, 2] - smaller_bbx[:, 2]) ** 2)
            loss += torch.sum(torch.abs(larger_bbx[:l, 1] ** 0.5 - smaller_bbx[:, 1] ** 0.5) + torch.abs(
                larger_bbx[:l, 2] ** 0.5 - smaller_bbx[:, 2] ** 0.5))
        return loss

    def calculate_loss(self, num_boxes, pred_bbx, gt_bbx):
        l = min(pred_bbx.size(0), gt_bbx.size(0))
        loss = (num_boxes - len(gt_bbx))**2
        if gt_bbx.size(0) <= pred_bbx.size(0):
            loss += self.weighted_sums(pred_bbx, gt_bbx, l)
        else:
            loss += self.weighted_sums(gt_bbx, pred_bbx, l)

        # loss = -box_iou(pred_bbx[:, 1:5], gt_bbx[:, 1:5])
        loss = torch.sum(loss)
        return loss

    def step(self, batch):
        x = batch[0]
        y = batch[1]
        y_hat = self.forward(x)

        #y_hat[0][0].size(0) == 0
        loss = torch.sum(torch.stack([self.calculate_loss(num_boxes=y_hat[1][idx] , pred_bbx=y_hat[0][idx], gt_bbx=bbx) for idx, bbx in enumerate(y)]), dim=0)
        loss = loss / len(y)

        step_outputs = {
            "loss": loss
        }
        self.log("step_loss", loss, prog_bar=True, logger=True)
        return step_outputs

    def training_step(self, batch, batch_idx):
        step_outputs = self.step(batch)
        return step_outputs

    def validation_step(self, batch, batch_idx):
        step_outputs = self.step(batch)
        return step_outputs

    def format_metrics(self, epoch_outputs, training=True):
        epoch_metrics = {}
        loss = torch.mean(torch.tensor([e["loss"] for e in epoch_outputs]))
        epoch_metrics["loss"] = loss
        self.log("loss", loss, prog_bar=True, logger=True)
        print(f"\nEpoch: {self.current_epoch}, lr: {self.opt.param_groups[0]['lr']}", end=" ")
        print(f"{'Training' if training else 'Validation'}, loss: {epoch_metrics['loss']:5.3f}", end=" ")
        out = Path("out.log")
        with out.open("a") as fp:
            fp.write(f"\nEpoch: {self.current_epoch}, lr: {self.opt.param_groups[0]['lr']}")
            fp.write(f"{'Training' if training else 'Validation'}, loss: {epoch_metrics['loss']:5.3f}")
        return epoch_metrics

    def training_epoch_end(self, training_epoch_outputs):
        self.format_metrics(training_epoch_outputs, training=True)

    def validation_epoch_end(self, validation_epoch_outputs):
        self.format_metrics(validation_epoch_outputs, training=False)