import numpy as np
import torch

def yolo_loss2(pred_fm, gt_fm):
    num_of_patches = pred_fm.shape[1]
    pred_fm = pred_fm.reshape(5, -1)
    if torch.nansum(pred_fm):
        pred_fm = torch.nan_to_num(pred_fm, nan=0.1)
    gt_fm = gt_fm.reshape(5, -1)

    # gt_bbx = convert_bbx_to_xyxy(gt_fm[1:])
    # pred_bbx = convert_bbx_to_xyxy(pred_fm[1:])

    gt_conf = gt_fm[0]
    pred_conf = pred_fm[0]

    object_in_cell = gt_conf
    empty_cell = 1 - gt_conf
    coord_weight = 3
    no_object_weight = 1
    # no_object_weight = 1 / (5 * num_of_patches)
    # no_object_weight = 1 / num_of_patches ** (np.log(num_of_patches) / 5)
    positive_indices = pi = torch.where(gt_fm[0] > 0)[0]
    num_of_negatives_to_use = min(int(num_of_patches**0.5) * len(pi) + 1, (num_of_patches ** 2) - 1)
    ni = 0
    a = torch.where(gt_conf == 0)[0].cpu().numpy()

    xy_loss = 0
    wh_loss = 0
    if pi.size(0) > 0:
        gt_x, gt_y = gt_fm[1:3, pi]
        pred_y, pred_x = pred_fm[1:3, pi]
        gt_w, gt_h = gt_fm[3:5, pi]
        pred_w, pred_h = pred_fm[3:5, pi]

        xy_loss = coord_weight * object_in_cell[pi] * ((gt_x - pred_x) ** 2 + (gt_y - pred_y) ** 2)
        wh_loss = coord_weight * object_in_cell[pi] * (
                    (gt_w ** 0.5 - pred_w ** 0.5) ** 2 + (gt_h ** 0.5 - pred_h ** 0.5) ** 2)

    # conf_loss = (object_in_cell + empty_cell * no_object_weight) * (gt_conf - pred_conf) ** 2
    pos_conf_loss = torch.sum((gt_conf[pi] - pred_conf[pi]) ** 2)
    if len(a) > 0:
        negative_indices = ni = np.random.choice(a, num_of_negatives_to_use)
        neg_conf_loss = torch.sum((gt_conf[ni] - pred_conf[ni]) ** 2)
        conf_loss = pos_conf_loss + neg_conf_loss
    elif pi.size(0) > 0:
        conf_loss = pos_conf_loss
    else:
        conf_loss = 0
    # conf_loss = (object_in_cell + empty_cell * no_object_weight) * (gt_conf - pred_conf) ** 2

    loss = torch.sum(xy_loss + wh_loss + conf_loss)
    if torch.isnan(loss):
        print(loss)
    # loss = nn.L1Loss()(pred_fm, gt_fm)
    return loss


def yolo_loss(pred_fm, gt_fm, epoch):
    num_of_patches = pred_fm.shape[1]
    pred_fm = pred_fm.reshape(5, -1)
    if torch.nansum(pred_fm):
        pred_fm = torch.nan_to_num(pred_fm, nan=0.1)
    gt_fm = gt_fm.reshape(5, -1)

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
    # no_object_weight = 1
    # no_object_weight = 1
    no_object_weight = max(1, epoch/10) / (4*num_of_patches)
    # no_object_weight = 1 / (num_of_patches ** 2)
    # no_object_weight = 1 / (num_of_patches ** 0.5)
    # no_object_weight = 1 / num_of_patches ** (np.log(num_of_patches))

    xy_loss = coord_weight * object_in_cell * ((gt_x - pred_x) ** 2 + (gt_y - pred_y) ** 2)
    wh_loss = coord_weight * object_in_cell * ((gt_w ** 0.5 - pred_w ** 0.5) ** 2 + (gt_h ** 0.5 - pred_h ** 0.5) ** 2)

    # wh_loss = coord_weight * object_in_cell * ((gt_w - pred_w) ** 2 + (gt_h - pred_h) ** 2)
    conf_loss = (object_in_cell + empty_cell * no_object_weight) * (gt_conf - pred_conf) ** 2

    loss = torch.sum(xy_loss + wh_loss + conf_loss)
    if torch.isnan(loss):
        print(loss)
    # loss = nn.L1Loss()(pred_fm, gt_fm)
    return loss
