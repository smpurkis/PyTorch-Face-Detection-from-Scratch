import torch
import torch.nn as nn
from PIL import ImageDraw
from torchvision import transforms
from torchvision.ops import nms


class ReduceSSDBoundingBoxes(nn.Module):
    def __init__(self, probability_threshold: float = 0.9, iou_threshold: float = 0.5, input_shape=(3, 320, 240),
                 patch_sizes=(60, 30, 15, 7), priors=None):
        super().__init__()
        self.probability_threshold = probability_threshold
        self.iou_threshold = iou_threshold
        self.input_shape = input_shape
        _, self.width, self.height = input_shape
        self.patch_sizes = patch_sizes
        self.multiply_priors = torch.unsqueeze(
            torch.cat([torch.tensor(1 / ps).repeat(ps * ps) for ps in self.patch_sizes]), dim=1)
        if priors is not None:
            self.priors = priors
        else:
            self.priors = self.calculate_priors()

    def calculate_priors(self):
        priors_list = []
        for ps in self.patch_sizes:
            priors = torch.zeros((4, ps, ps))
            i, j = torch.where(priors[0] >= 0)
            priors[0, i, j] = priors[0, i, j] + 1 / ps * i
            priors[1, i, j] = priors[1, i, j] + 1 / ps * j
            priors[2, i, j] = priors[2, i, j]
            priors[3, i, j] = priors[3, i, j]
            priors = priors.permute(1, 2, 0).reshape(ps * ps, 4)
            priors_list.append(priors)
        priors = torch.cat(priors_list, dim=0)
        return priors

    def remove_low_probabilty_bbx(self, x):
        i = torch.where(x[:, 0] > self.probability_threshold)[0]
        if i.shape[0] == 0:
            return torch.empty([0]), torch.tensor(0)
        bbx = x[i, :]
        return bbx, torch.tensor(1)

    def scale_batch_bbx_xywh(self, x):
        mask = torch.where(x[:, 0] > self.probability_threshold)[0]
        x = x.float()
        scaled_x = torch.clone(x).float()
        scaled_x[mask, 1:3] = scaled_x[mask, 1:3] * torch.unsqueeze(self.multiply_priors[mask], dim=0)
        scaled_x[mask, 1:5] = scaled_x[mask, 1:5] + torch.unsqueeze(self.priors[mask], dim=0)
        scaled_x[:, [1, 3]] = scaled_x[:, [1, 3]] * self.width
        scaled_x[:, [2, 4]] = scaled_x[:, [2, 4]] * self.height
        return scaled_x

    def convert_batch_to_xywh(self, x):
        x[:, 3] = x[:, 3] - x[:, 1]
        x[:, 4] = x[:, 4] - x[:, 2]
        return x

    def convert_batch_to_xyxy(self, x):
        x[:, 3] = x[:, 3] + x[:, 1]
        x[:, 4] = x[:, 4] + x[:, 2]
        return x

    def forward(self, x):
        x = self.scale_batch_bbx_xywh(x)
        x, boxes_exist = self.remove_low_probabilty_bbx(x)
        if boxes_exist == 1:
            x = self.convert_batch_to_xyxy(x)
            bbx = torch.round(x[:, 1:])
            scores = x[:, 0]
            breakpoint()
            bbxis = nms(boxes=bbx, scores=scores, iou_threshold=self.iou_threshold)
            x = torch.cat([scores.view(-1, 1), bbx], dim=1)
            out = self.convert_batch_to_xywh(x[bbxis])
            return out
        else:
            return torch.empty(0).reshape(0, 5)


class ReduceBoundingBoxes(nn.Module):
    def __init__(self, probability_threshold: float = 0.9, iou_threshold: float = 0.5, input_shape=(3, 320, 240),
                 num_of_patches=40):
        super().__init__()
        self.probability_threshold = probability_threshold
        self.iou_threshold = iou_threshold
        self.input_shape = input_shape
        _, self.width, self.height = input_shape
        self.x_patch_size = self.width / num_of_patches
        self.y_patch_size = self.height / num_of_patches

    def remove_low_probabilty_bbx(self, x):
        i, j = torch.where(x[0] > self.probability_threshold)
        if i.shape[0] == 0 and j.shape[0] == 0:
            return torch.empty([0]), torch.tensor(0)
        bbx = x[:, i, j].permute(1, 0)
        return bbx, torch.tensor(1)

    def scale_batch_bbx_xywh(self, x):
        i, j = torch.where(x[0] > self.probability_threshold)
        x = x.float()
        scaled_x = torch.clone(x).float()
        scaled_x[1, i, j] = x[1, i, j] * self.x_patch_size + i * self.x_patch_size
        scaled_x[2, i, j] = x[2, i, j] * self.y_patch_size + j * self.y_patch_size
        scaled_x[3, i, j] = x[3, i, j] * self.width
        scaled_x[4, i, j] = x[4, i, j] * self.height
        return scaled_x

    def scale_batch_bbx(self, x):
        i, j = torch.where(x[0] > self.probability_threshold)
        scaled_x = torch.clone(x).float()
        scaled_x[1, i, j] = x[1, i, j] * self.x_patch_size + i * self.x_patch_size
        scaled_x[2, i, j] = x[2, i, j] * self.y_patch_size + j * self.y_patch_size
        scaled_x[3, i, j] = x[3, i, j] * self.x_patch_size + i * self.x_patch_size
        scaled_x[4, i, j] = x[4, i, j] * self.y_patch_size + j * self.y_patch_size
        return scaled_x

    def convert_batch_to_xywh(self, x):
        x[:, 3] = x[:, 3] - x[:, 1]
        x[:, 4] = x[:, 4] - x[:, 2]
        return x

    def convert_batch_to_xyxy(self, x):
        x[:, 3] = x[:, 3] + x[:, 1]
        x[:, 4] = x[:, 4] + x[:, 2]
        return x

    def forward(self, x):
        x = self.scale_batch_bbx_xywh(x)
        x, boxes_exist = self.remove_low_probabilty_bbx(x)
        if boxes_exist == 1:
            x = self.convert_batch_to_xyxy(x)
            bbx = torch.round(x[:, 1:])
            scores = x[:, 0]
            bbxis = nms(boxes=bbx, scores=scores, iou_threshold=self.iou_threshold)
            x = torch.cat([scores.view(-1, 1), bbx], dim=1)
            out = self.convert_batch_to_xywh(x[bbxis])
            # out = x[bbxis]
            return out
        else:
            return torch.empty(0).reshape(0, 5)


def convert_bbx_to_xyxy(bbx):
    return bbx[0], bbx[1], bbx[0] + bbx[2], bbx[1] + bbx[3]


def draw_bbx(img, bbx, input_shape=(320, 240), save_name="image", show=False):
    bbxs = bbx
    if isinstance(bbxs, torch.Tensor):
        if len(bbxs.shape) == 3:
            num_of_patches = bbxs.shape[1]
            reduce_bounding_boxes = ReduceBoundingBoxes(0.9, 0.5, (3, *input_shape), num_of_patches)
            bbxs = reduce_bounding_boxes(bbxs)
    elif isinstance(bbxs, list):
        pass
    if isinstance(img, torch.Tensor):
        img = transforms.ToPILImage()(img)
    draw = ImageDraw.Draw(img)
    for b in bbxs:
        if len(b) == 5:
            b = b[1:]
        if b[2] <= 15 or b[3] <= 15:
            width = 1
        else:
            width = 3
        bbx = convert_bbx_to_xyxy(b)
        draw.rectangle(bbx, outline="blue", width=width)
        # bbx = b[1:]
        # draw.rectangle(bbx.detach().cpu().numpy(), outline="blue", width=width)
    if show:
        img.show()
    else:
        img.save(f"imgs/{save_name}.png")
    return draw
