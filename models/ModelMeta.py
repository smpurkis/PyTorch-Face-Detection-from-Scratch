from pathlib import Path

import torch
from pytorch_lightning import LightningModule
from torch.optim._multi_tensor import SGD, Adam
from torchvision.ops import box_iou

from datasets.utils import draw_bbx
from losses.YoloLoss import yolo_loss


class SAMSGD(Adam):
    """ SGD wrapped with Sharp-Aware Minimization
    Args:
        params: tensors to be optimized
        lr: learning rate
        momentum: momentum factor
        dampening: damping factor
        weight_decay: weight decay factor
        nesterov: enables Nesterov momentum
        rho: neighborhood size
    """

    def __init__(self,
                 params,
                 lr: float,
                 rho: float = 0.05,
                 *args,
                 **kwargs,
                 ):
        if rho <= 0:
            raise ValueError(f"Invalid neighborhood size: {rho}")
        super().__init__(params, lr, *args, **kwargs)
        if len(self.param_groups) > 1:
            raise ValueError("Not supported")
        self.param_groups[0]["rho"] = rho

    @torch.no_grad()
    def step(self,
             closure
             ) -> torch.Tensor:
        """
        Args:
            closure: A closure that reevaluates the model and returns the loss.
        Returns: the loss value evaluated on the original point
        """
        closure = torch.enable_grad()(closure)
        loss = closure().detach()

        for group in self.param_groups:
            grads = []
            params_with_grads = []

            rho = group['rho']
            # update internal_optim's learning rate

            for p in group['params']:
                if p.grad is not None:
                    # without clone().detach(), p.grad will be zeroed by closure()
                    grads.append(p.grad.clone().detach())
                    params_with_grads.append(p)
            device = grads[0].device

            # compute \hat{\epsilon}=\rho/\norm{g}\|g\|
            grad_norm = torch.stack([g.detach().norm(2).to(device) for g in grads]).norm(2)
            epsilon = grads  # alias for readability
            torch._foreach_mul_(epsilon, rho / grad_norm)

            # virtual step toward \epsilon
            torch._foreach_add_(params_with_grads, epsilon)
            # compute g=\nabla_w L_B(w)|_{w+\hat{\epsilon}}
            closure()
            # virtual step back to the original point
            torch._foreach_sub_(params_with_grads, epsilon)

        super().step()
        return loss


class ModelMeta(LightningModule):
    def __init__(self, model, lr=1e-4, pretrained=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.lr = lr
        self.automatic_optimization = False

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        # optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        optimizer = SAMSGD(self.parameters(), lr=self.lr)
        self.opt = optimizer
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50], gamma=0.1)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200*100], gamma=0.1)
        # return [optimizer], [scheduler]
        return optimizer

    def step(self, batch, batch_idx, validation=False):
        x, y, gt_bbx = batch

        # y_hat = self.forward(x)
        # loss = 0

        def closure():
            self.opt.zero_grad()
            closure_loss = 0
            y_hat = self.forward(x)
            for i in range(y.shape[0]):
                predicted_boxes = y_hat[i]
                ground_truth_boxes = y[i]
                closure_loss += yolo_loss(predicted_boxes, ground_truth_boxes)
            closure_loss = closure_loss / len(y)
            closure_loss.backward()
            return closure_loss

        loss = self.opt.step(closure)
        self.scheduler.step()

        with torch.no_grad():
            x, y, gt_bbx = batch
            y_hat = self.forward(x)
            loss = 0

            if batch_idx == 0 and self.current_epoch % 2 == 0:
                # gt_bbx_check = self.model.non_max_suppression(y)
                pred_bbx = self.model.non_max_suppression(y_hat)
                test_img = x[0]
                test_gt = gt_bbx[0]
                test_pred = pred_bbx[0]
                draw_bbx(
                    img=test_img,
                    bbx=test_pred,
                    input_shape=self.model.input_shape,
                    save_name=f"{'validation' if validation else 'train'}_epoch_{self.current_epoch}"
                )
                # draw_bbx(
                #     img=test_img,
                #     bbx=test_gt,
                #     input_shape=self.model.input_shape,
                #     save_name=f"{'validation' if validation else 'train'}_epoch_{self.current_epoch}"
                # )
                i = 0
            total_iou = 0.0
            total_recall = 0.0
            total_precision = 0.0
            for i in range(y.shape[0]):
                predicted_boxes = y_hat[i]
                ground_truth_boxes = y[i]
                loss += yolo_loss(predicted_boxes, ground_truth_boxes)

                gt_bbx = self.model.reduce_bounding_boxes(ground_truth_boxes).reshape(5, -1)[1:].permute(1, 0).to(
                    ground_truth_boxes.device)
                pred_bbx = self.model.reduce_bounding_boxes(predicted_boxes)

                if pred_bbx.shape[0] > 0:
                    pred_bbx = pred_bbx[:, 1:]
                    gt_bbx[:, 2] = gt_bbx[:, 2] + gt_bbx[:, 0]
                    gt_bbx[:, 3] = gt_bbx[:, 3] + gt_bbx[:, 1]

                    pred_bbx[:, 2] = pred_bbx[:, 2] + pred_bbx[:, 0]
                    pred_bbx[:, 3] = pred_bbx[:, 3] + pred_bbx[:, 1]
                    iou = torch.nan_to_num(box_iou(gt_bbx, pred_bbx), 0)
                    if gt_bbx.shape[0] == 0:
                        recall = 1.0 if pred_bbx.shape[0] == 0 else 0.0
                    else:
                        recall = torch.where(iou > 0.5)[0].shape[0] / gt_bbx.shape[0]
                    total_recall += recall
                    precision = torch.where(iou > 0.5)[0].shape[0] / pred_bbx.shape[0]
                    total_precision += precision
                    total_iou += torch.sum(iou)
            # loss = loss / len(y)
            total_recall = total_recall / len(y)
            total_precision = total_precision / len(y)
            total_iou = total_iou / len(y)

            step_outputs = {
                "loss": loss,
                "total_iou": total_iou,
                "total_recall": total_recall,
                "total_precision": total_precision,
            }
            self.log("step_loss", loss, prog_bar=True, logger=True, on_step=True)
        return step_outputs

    def training_step(self, batch, batch_idx):
        step_outputs = self.step(batch, batch_idx)
        return step_outputs

    def validation_step(self, batch, batch_idx):
        step_outputs = self.step(batch, batch_idx, validation=True)
        return step_outputs

    def format_metrics(self, epoch_outputs, training=True):
        step_str = 'training' if training else 'validation'
        epoch_metrics = {}
        loss = torch.mean(torch.tensor([e["loss"] for e in epoch_outputs]))
        epoch_metrics["loss"] = loss
        total_iou = torch.mean(torch.tensor([e["total_iou"] for e in epoch_outputs]))
        epoch_metrics["total_iou"] = total_iou
        total_recall = torch.mean(torch.tensor([e["total_recall"] for e in epoch_outputs]))
        epoch_metrics["total_recall"] = total_recall
        total_precision = torch.mean(torch.tensor([e["total_precision"] for e in epoch_outputs]))
        epoch_metrics["total_precision"] = total_precision
        f1_score = 2 * total_precision * total_recall / (total_precision + total_recall)
        epoch_metrics["f1_score"] = f1_score
        self.log("loss", loss, prog_bar=True, logger=True, on_epoch=True)
        self.log(f"{step_str} loss", loss, prog_bar=True, logger=True, on_epoch=True)
        self.log(f"{step_str} iou", epoch_metrics['loss'], prog_bar=True, logger=True, on_epoch=True)
        self.log(f"{step_str} recall", epoch_metrics['total_recall'], prog_bar=True, logger=True, on_epoch=True)
        print(f"\nEpoch: {self.current_epoch}, lr: {self.opt.param_groups[0]['lr']}", end=" ")
        print(f"\n{step_str}, loss: {epoch_metrics['loss']:5.3f}", end=" ")

        if not training:
            self.epoch_metrics = epoch_metrics
        else:
            out = Path("out.log")
            with out.open("a") as fp:
                fp.write(f"\nEpoch: {self.current_epoch}, lr: {self.opt.param_groups[0]['lr']} ")
                fp.write(f"training, loss: {epoch_metrics['loss']:5.3f}, iou: {epoch_metrics['total_iou']:5.3f},"
                         f"recall {epoch_metrics['total_recall']:5.3f}, precision {epoch_metrics['total_precision']:5.3f}"
                         f", f1_score {epoch_metrics['f1_score']:5.3f} ")
                fp.write(
                    f"validation, loss: {self.epoch_metrics['loss']:5.3f}, iou: {self.epoch_metrics['total_iou']:5.3f},"
                    f" recall {self.epoch_metrics['total_recall']:5.3f}, precision {self.epoch_metrics['total_precision']:5.3f}"
                    f", f1_score {self.epoch_metrics['f1_score']:5.3f} ")
        return epoch_metrics

    def training_epoch_end(self, training_epoch_outputs):
        self.format_metrics(training_epoch_outputs, training=True)

    def validation_epoch_end(self, validation_epoch_outputs):
        self.format_metrics(validation_epoch_outputs, training=False)
