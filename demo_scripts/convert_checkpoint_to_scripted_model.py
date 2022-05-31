import os

import cv2
import torch

from datasets.utils import convert_bbx_to_xyxy, ReduceBoundingBoxes
from models import ModelMeta
from models.MobilenetV3Backbone import MobilenetV3Backbone
from models.PoolResnet import PoolResnet
from models.Resnet import Resnet

os.environ["CUDA_VISIBLE_DEVICES"] = ""

num_of_patches = 10
input_shape = (480, 480)

# checkpoint = torch.load("lightning_logs/custom_poolresnet_32_10x10_480x480_sam_adam/checkpoints/epoch=69-step=56279.ckpt",
#                         map_location=torch.device("cpu"))
# checkpoint = torch.load("lightning_logs/custom_poolresnet_64_10x10_480x480_sam_adam/checkpoints/epoch=69-step=56279.ckpt",
#                         map_location=torch.device("cpu"))
# checkpoint = torch.load("lightning_logs/custom_poolresnet_128_10x10_480x480_sam_adam_all_data/checkpoints/epoch=69-step=112699.ckpt",
#                         map_location=torch.device("cpu"))
# checkpoint = torch.load("lightning_logs/custom_resnet_64_15x15_480x480_sam_adam/checkpoints/epoch=52-step=42611.ckpt",
#                         map_location=torch.device("cpu"))
# checkpoint = torch.load(
#     "lightning_logs/pretrained_mobilenetv3backbone_576_15x15_480x480_sam_adam_all_data/checkpoints/epoch=69-step=112699.ckpt",
#     map_location=torch.device("cpu"),
# )


def define_model_from_checkpoint(checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    model = PoolResnet(
        filters=128,
        input_shape=(3, *input_shape),
        num_of_patches=num_of_patches,
    )
    model_setup = ModelMeta(model=model, lr=1e-4)

    model_setup.load_state_dict(checkpoint["state_dict"])

    model.num_of_patches = num_of_patches
    model.reduce_bounding_boxes = ReduceBoundingBoxes(
        probability_threshold=0.7,
        iou_threshold=0.01,
        input_shape=model.input_shape,
        num_of_patches=model.num_of_patches,
    )
    model = model.eval()
    model.summary()
    model = torch.jit.script(model)
    model._save_for_lite_interpreter(
        "./saved_models/custom_poolresnet_128_10x10_480x480_sam_adam_all_data.pth"
    )
    return model


def define_model(model_path: str):
    model = torch.jit.load(model_path)
    return model


if __name__ == "__main__":
    model = define_model_from_checkpoint(
        "lightning_logs/custom_poolresnet_128_10x10_480x480_sam_adam_all_data/checkpoints/epoch=69-step=112699.ckpt"
    )
