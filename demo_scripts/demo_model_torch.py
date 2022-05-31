import os

import cv2
import torch

from datasets.utils import convert_bbx_to_xyxy, ReduceBoundingBoxes
from models import ModelMeta
from models.MobilenetV3Backbone import MobilenetV3Backbone
from models.PoolResnet import PoolResnet
from models.Resnet import Resnet

os.environ['CUDA_VISIBLE_DEVICES'] = ""

num_of_patches = 10
input_shape = (480, 480)

model = PoolResnet(
    filters=128,
    input_shape=(3, *input_shape),
    num_of_patches=num_of_patches,
)
model_setup = ModelMeta(
    model=model,
    lr=1e-4
)

checkpoint = torch.load("lightning_logs/custom_poolresnet_128_10x10_480x480_sam_adam_all_data/checkpoints/epoch=69-step=112699.ckpt",
                        map_location=torch.device("cpu"))
# checkpoint = torch.load("lightning_logs/custom_poolresnet_64_10x10_480x480_sam_adam/checkpoints/epoch=69-step=56279.ckpt",
#                         map_location=torch.device("cpu"))
# checkpoint = torch.load("lightning_logs/custom_resnet_64_15x15_480x480_sam_adam/checkpoints/epoch=52-step=42611.ckpt",
#                         map_location=torch.device("cpu"))
# checkpoint = torch.load("lightning_logs/pretrained_mobilenetv3backbone_576_15x15_480x480_sam_adam_all_data/checkpoints/epoch=69-step=112699.ckpt",
#                         map_location=torch.device("cpu"))
model_setup.load_state_dict(checkpoint["state_dict"])

model.num_of_patches = num_of_patches
model.reduce_bounding_boxes = ReduceBoundingBoxes(
            probability_threshold=0.7,
            iou_threshold=0.01,
            input_shape=model.input_shape,
            num_of_patches=model.num_of_patches
        )
model = model.eval()
model.summary()
# model = torch.jit.trace(model, (torch.rand(1, 3, 480, 480), torch.tensor(1)))
model = torch.jit.script(model)
# model._save_for_lite_interpreter("./saved_models/custom_poolresnet_128_10x10_480x480_sam_adam_all_data.pth")
# model = torch.jit.load("custom_poolresnet_128_10x10_480x480_sam_adam.onnx")
# t = torch.from_numpy(cv2.cvtColor(cv2.imread("/home/sam/PycharmProjects/python/PyTorch-Face-Detection-from-Scratch/imgs/validation_epoch_0.png"), cv2.COLOR_BGR2RGB)).permute(2, 0, 1).unsqueeze(0) / 255.
# o = model(t)
# torch.onnx.export(
#     model_setup,
#     t,
#     "custom_poolresnet_128_10x10_480x480_sam_adam.pth.onnx",
#     opset_version=13,
#     verbose=True,
#     dynamic_axes={"output": {0: "#boxes"}}
# )

@torch.no_grad()
def extract_face(frame):
    image = cv2.resize(frame, (480, 480))
    tensor = torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
    tensor = torch.stack([tensor, tensor], dim=0)
    bbxs = model(tensor, predict=torch.tensor(1))
    for b in bbxs:
        if len(b) == 5:
            b = b[1:]
        if b[2] <= 15 or b[3] <= 15:
            width = 1
        else:
            width = 3
        bbx = [int(p.numpy()) for p in convert_bbx_to_xyxy(b)]
        image = cv2.rectangle(image, pt1=(bbx[0], bbx[1]), pt2=(bbx[2], bbx[3]), thickness=width, color=(0, 0, 200))
    return image


cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    process_frame = extract_face(frame)
    cv2.imshow('Input', process_frame)

    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()
