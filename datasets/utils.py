import torch
import torch.nn as nn
from PIL import ImageDraw
from torchvision import transforms
from torchvision.ops import nms


class ReduceBoundingBoxes(nn.Module):
    def __init__(self, probability_threshold: float = 0.9, iou_threshold: float = 0.5, input_shape=(3, 320, 240),
                 num_of_patches=40):
        super().__init__()
        self.probability_threshold = probability_threshold
        self.iou_threshold = iou_threshold
        self.input_shape = input_shape
        _, self.width, self.height = input_shape
        # print("self.width, self.height", self.width, self.height)
        self.x_patch_size = self.width / num_of_patches
        self.y_patch_size = self.height / num_of_patches
        # print("x_patch_size 2, y_patch_size 2", self.x_patch_size, self.y_patch_size)

    def remove_low_probabilty_bbx(self, x):
        i, j = torch.where(x[0] > self.probability_threshold)
        if len(i) == 0 and len(j) == 0:
            return torch.empty([0]), False
        bbx = x[:, i, j].permute(1, 0)
        # bbx[:, 2] = bbx[:, 0] + bbx[:, 2]
        # bbx[:, 3] = bbx[:, 1] + bbx[:, 3]
        return bbx, True

    def convert_batch_bbx_to_xyxy_scaled(self, x):
        i, j = torch.where(x[0] > self.probability_threshold)
        scaled_x = torch.clone(x).float()
        scaled_x[1, i, j] = x[1, i, j] * self.x_patch_size + i * self.x_patch_size
        scaled_x[2, i, j] = x[2, i, j] * self.y_patch_size + j * self.y_patch_size
        scaled_x[3, i, j] = x[3, i, j] * self.width + i * self.x_patch_size
        scaled_x[4, i, j] = x[4, i, j] * self.height + j * self.y_patch_size
        return scaled_x

    def convert_batch_to_xywh(self, x):
        x[:, 3] = x[:, 3] - x[:, 1]
        x[:, 4] = x[:, 4] - x[:, 2]
        return x

    def convert_bbx_to_xyxy(self, x, i, j):
        return torch.tensor(
            [x[0],
             x[1] * self.width + i * self.x_patch_size,
             x[2] * self.height + j * self.y_patch_size,
             x[3] * self.width + i * self.x_patch_size,
             x[4] * self.height + j * self.y_patch_size]
        )

    def forward(self, x):
        # x2 = torch.clone(x)
        # x2 = x
        # for i in range(x2.shape[1]):
        #     for j in range(x2.shape[2]):
        #         if x2[0, i, j] > 0:
        #             x2[:, i, j] = self.convert_bbx_to_xyxy(x2[:, i, j], i, j)
        x = self.convert_batch_bbx_to_xyxy_scaled(x) # tensor([[  1.0000, 114.3750,  40.4598, 215.3125, 144.0974]])
        x, boxes_exist = self.remove_low_probabilty_bbx(x)
        if boxes_exist:
            bbx = torch.round(x[:, 1:])
            scores = x[:, 0]
            bbxis = nms(boxes=bbx, scores=scores, iou_threshold=self.iou_threshold)
            x = torch.cat([scores.view(-1, 1), bbx], dim=1)
            out = self.convert_batch_to_xywh(x[bbxis])
            return out
        else:
            return torch.empty(0)


def convert_bbx_to_xyxy(bbx):
    return bbx[0], bbx[1], bbx[0] + bbx[2], bbx[1] + bbx[3]


def draw_bbx(img, bbx, input_shape=(320, 240), save_name="image"):
    if len(bbx.shape) == 3:
        num_of_patches = bbx.shape[1]
        reduce_bounding_boxes = ReduceBoundingBoxes(0.9, 0.5, (3, *input_shape), num_of_patches)
        bbx = reduce_bounding_boxes(bbx)
    if isinstance(img, torch.Tensor):
        img = transforms.ToPILImage()(img)
    draw = ImageDraw.Draw(img)
    for b in bbx:
        if b[3] <= 15 or b[4] <= 15:
            width = 1
        else:
            width = 3
        draw.rectangle(convert_bbx_to_xyxy(b[1:]), outline="blue", width=width)
    # img.show()
    img.save(f"imgs/{save_name}.png")
    return draw
