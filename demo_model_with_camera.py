import os

import cv2
import torch

from datasets.utils import convert_bbx_to_xyxy, ReduceBoundingBoxes
from models import ModelMeta
from models.Resnet import Resnet

os.environ['CUDA_VISIBLE_DEVICES'] = ""

num_of_patches = 10
input_shape = (480, 480)

model = Resnet(
    filters=32,
    input_shape=(3, *input_shape),
    num_of_patches=num_of_patches,
    num_of_residual_blocks=10
)
model_setup = ModelMeta(
    model=model,
    lr=1e-4
)

checkpoint = torch.load("lightning_logs/custom_32_10x10_sam_adam/checkpoints/epoch=69-step=56279.ckpt",
                        map_location=torch.device("cpu"))
model_setup.load_state_dict(checkpoint["state_dict"])
model.num_of_patches = num_of_patches
model.reduce_bounding_boxes = ReduceBoundingBoxes(
            probability_threshold=model.probability_threshold,
            iou_threshold=model.iou_threshold,
            input_shape=model.input_shape,
            num_of_patches=model.num_of_patches
        )
model = model.eval()
model = torch.jit.script(model)

# torch.save(model.state_dict(), "test_model")

# test = torch.rand(3, 480, 480)
# test_image = PIL.Image.open("/home/sam/PycharmProjects/python/PyTorch-Face-Detection-from-Scratch/imgs/train_epoch_0.png")

# test_image_path = "/home/sam/PycharmProjects/python/PyTorch-Face-Detection-from-Scratch/imgs/train_epoch_0.png"
# test_image_path = "/home/sam/PycharmProjects/python/PyTorch-Face-Detection-from-Scratch/imgs/validation_epoch_0.png"
# # test_image_path = "test_image.jpeg"
# test_image_torch = torch.from_numpy(cv2.cvtColor(cv2.imread(test_image_path), cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
# #
# with torch.no_grad():
#     # model.reduce_bounding_boxes = ReduceBoundingBoxes(0.5, 0.5, (3, 480, 480), 10)
#     model.reduce_bounding_boxes = ReduceBoundingBoxes(model.probability_threshold, model.iou_threshold, model.input_shape, 10)
#     out = model.predict(test_image_torch, probability_threshold=0.8, iou_threshold=0.1)
#     draw_bbx(test_image_torch, out, input_shape=input_shape, show=True)

@torch.no_grad()
def extract_face(frame):
    tensor = torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
    bbxs = model(tensor)[0]
    # image, bbxs = model.predict(tensor, probability_threshold=0.8, iou_threshold=0.1)
    # image = 255.*image.permute(1, 2, 0).numpy()
    # image = image.astype(np.uint8)
    image = cv2.resize(frame, (480, 480))
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
