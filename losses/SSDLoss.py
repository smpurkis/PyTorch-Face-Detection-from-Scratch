import torch


def ssd_loss(pred_fm, gt_fm):
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

    xy_loss = coord_weight * object_in_cell * ((gt_x - pred_x) ** 2 + (gt_y - pred_y) ** 2)
    # xy_loss = 0
    wh_loss = coord_weight * object_in_cell * ((gt_w ** 0.5 - pred_w ** 0.5) ** 2 + (gt_h ** 0.5 - pred_h ** 0.5) ** 2)
    # wh_loss = 0
    conf_loss = (object_in_cell + empty_cell * no_object_weight) * (gt_conf - pred_conf) ** 2
    # conf_loss = 0

    loss = torch.sum(xy_loss + wh_loss + conf_loss)
    if torch.isnan(loss):
        print(loss)
    # loss = nn.L1Loss()(pred_fm, gt_fm)
    return loss
