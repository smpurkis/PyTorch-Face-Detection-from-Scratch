import torch
import torch.nn as nn
from PIL import ImageDraw
from torchvision import transforms
from torchvision.ops import nms

class SSDReduceBoundingBoxes(nn.Module):
    def __init__(self, probability_threshold: float = 0.9, iou_threshold: float = 0.5, input_shape=(3, 320, 240)):
        super().__init__()
        self.probability_threshold = probability_threshold
        self.iou_threshold = iou_threshold
        self.input_shape = input_shape
        _, self.width, self.height = input_shape

    def remove_low_probabilty_bbx(self, x):
        i, j = torch.where(x[0] > self.probability_threshold)
        if i.shape[0] == 0 and j.shape[0] == 0:
            return torch.empty([0]), torch.tensor(0)
        bbx = x[:, i, j].permute(1, 0)
        return bbx, torch.tensor(1)

    def scale_batch_bbx_xywh(self, x, num_of_patches):
        x_patch_size = self.width / num_of_patches
        y_patch_size = self.height / num_of_patches
        i, j = torch.where(x[0] > self.probability_threshold)
        x = x.float()
        scaled_x = torch.clone(x).float()
        scaled_x[1, i, j] = x[1, i, j] * x_patch_size + i * x_patch_size
        scaled_x[2, i, j] = x[2, i, j] * y_patch_size + j * y_patch_size
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

    def forward(self, outs, num_of_patches):
        z = []
        for i in range(len(num_of_patches)):
            x = self.scale_batch_bbx_xywh(outs[i], num_of_patches[i])
            x, boxes_exist = self.remove_low_probabilty_bbx(x)
            if boxes_exist == 1:
                z.append(x)
        if len(z) > 0:
            x = torch.cat(z, dim=0)
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

    def apply_priors(self, x):
        n, i, j = torch.where(x[:, 0] > self.probability_threshold)
        x = x.float()
        scaled_x = torch.clone(x).float()
        scaled_x[n, 1, i, j] = x[n, 1, i, j] * self.x_patch_size + i * self.x_patch_size
        scaled_x[n, 2, i, j] = x[n, 2, i, j] * self.y_patch_size + j * self.y_patch_size
        scaled_x[n, 3, i, j] = x[n, 3, i, j] * self.width
        scaled_x[n, 4, i, j] = x[n, 4, i, j] * self.height
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


@torch.no_grad()
def draw_bbx(img, bbx, input_shape=(320, 240), save_name="image", show=False):
    bbxs = bbx
    if isinstance(bbxs, torch.Tensor):
        if len(bbxs.shape) == 3:
            num_of_patches = bbxs.shape[1]
            reduce_bounding_boxes = ReduceBoundingBoxes(0.5, 0.5, (3, *input_shape), num_of_patches)
            bbxs = reduce_bounding_boxes(bbxs)
        # elif len(bbxs.shape) == 2:
        #     reduce_bounding_boxes = ReduceSSDBoundingBoxes(0.5, 0.5, input_shape)
        #     bbxs = reduce_bounding_boxes(bbxs)
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
