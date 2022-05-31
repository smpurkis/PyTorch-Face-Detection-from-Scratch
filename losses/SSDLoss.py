import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CustomBCELoss(torch.nn.Module):
    def __init__(self, pos_weight=1):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, input, target):
        epsilon = 10**-7
        input = input.clamp(epsilon, 1 - epsilon)
        my_bce_loss = torch.sum(
            -1
            * (
                self.pos_weight * target * torch.log(input)
                + (1 - target) * torch.log(1 - input)
            )
        )
        return my_bce_loss


def hard_negative_mining(loss, labels, neg_pos_ratio):
    """
    It used to suppress the presence of a large number of negative prediction.
    It works on image level not batch level.
    For any example/image, it keeps all the positive predictions and
     cut the number of negative predictions to make sure the ratio
     between the negative examples and positive examples is no more
     the given ratio for an image.

    Args:
        loss (N, num_priors): the loss for each example.
        labels (N, num_priors): the labels.
        neg_pos_ratio:  the ratio between the negative examples and positive examples.
    """
    pos_mask = labels > 0  # positive label mask for each image
    num_pos = pos_mask.long().sum(
        dim=1, keepdim=True
    )  # calculates the number of positive images for each image
    num_neg = (
        num_pos * neg_pos_ratio
    )  # calculates the number of negative images for each image

    loss[pos_mask] = -math.inf  # sets all the positive confidences to negative infinity
    _, indexes = loss.sort(
        dim=1, descending=True
    )  # order the losses and extract the indices in descending order per image
    _, orders = indexes.sort(dim=1)
    neg_mask = orders < num_neg
    return pos_mask | neg_mask


def ssd_loss(confidence, predicted_locations, labels, gt_locations, neg_pos_ratio):
    """Compute classification loss and smooth l1 loss.

    Args:
        confidence (batch_size, num_priors, num_classes): class predictions.
        locations (batch_size, num_priors, 4): predicted locations.
        labels (batch_size, num_priors): real labels of all the priors.
        boxes (batch_size, num_priors, 4): real boxes corresponding all the priors.
    """
    with torch.no_grad():
        # derived from cross_entropy=sum(log(p))
        # loss = -F.log_softmax(confidence, dim=1)[:, :, 0]
        # loss2 = torch.abs(labels - confidence)
        loss = -torch.log(confidence)
        mask = hard_negative_mining(loss, labels, neg_pos_ratio)

    confidence = confidence[mask]
    labels_masked = torch.round(labels[mask])

    # for some reason the builtin torch.nn.BCELoss errs with exception some autocast function
    loss_fn = CustomBCELoss()
    classification_loss = loss_fn(confidence, labels_masked)
    pos_mask = labels > 0
    predicted_locations = predicted_locations[pos_mask, :].reshape(-1, 4)
    gt_locations = gt_locations[pos_mask, :].reshape(-1, 4)
    smooth_l1_loss = F.smooth_l1_loss(
        predicted_locations, gt_locations, reduction="sum"
    )  # smooth_l1_loss
    # smooth_l1_loss = F.mse_loss(predicted_locations, gt_locations, reduction='sum')  #l2 loss
    num_pos = gt_locations.size(0)
    return (smooth_l1_loss + classification_loss) / num_pos


def ssd_loss2(pred_fm, gt_fm):
    num_of_predictions = pred_fm.shape[0]
    pred_fm = pred_fm.permute(1, 0)
    gt_fm = gt_fm.permute(1, 0)
    if torch.nansum(pred_fm):
        pred_fm = torch.nan_to_num(pred_fm, nan=0.1)
    pred_fm = torch.clamp(pred_fm, min=0.0, max=1.0)

    # gt_bbx = convert_bbx_to_xyxy(gt_fm[1:])
    # pred_bbx = convert_bbx_to_xyxy(pred_fm[1:])

    gt_conf = gt_fm[0]
    pred_conf = pred_fm[0]
    gt_x, gt_y = gt_fm[[1, 2]]
    pred_y, pred_x = pred_fm[[1, 2]]
    gt_w, gt_h = gt_fm[[3, 4]]
    pred_w, pred_h = pred_fm[[3, 4]]

    object_in_cell = gt_conf
    empty_cell = 1 - gt_conf
    coord_weight = 3
    # no_object_weight = 0
    # no_object_weight = 1
    # no_object_weight = 1 / (2 * num_of_predictions)
    no_object_weight = 1 / num_of_predictions**1

    xy_loss = (
        coord_weight * object_in_cell * ((gt_x - pred_x) ** 2 + (gt_y - pred_y) ** 2)
    )
    # xy_loss = 0
    wh_loss = (
        coord_weight
        * object_in_cell
        * ((gt_w**0.5 - pred_w**0.5) ** 2 + (gt_h**0.5 - pred_h**0.5) ** 2)
    )
    # wh_loss = 0
    conf_loss = (object_in_cell + empty_cell * no_object_weight) * (
        gt_conf - pred_conf
    ) ** 2
    # conf_loss = 0

    loss = torch.sum(xy_loss + wh_loss + conf_loss)
    if torch.isnan(loss):
        print(loss)
    # loss = nn.L1Loss()(pred_fm, gt_fm)
    return loss
