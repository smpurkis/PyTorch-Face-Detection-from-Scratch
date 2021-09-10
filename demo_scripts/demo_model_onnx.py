import os

import cv2
import numpy as np
import onnxruntime as rt
import onnx
from pathlib import Path

from datasets.utils import convert_bbx_to_xyxy

onnx_path = Path("../saved_models/custom_poolresnet_128_10x10_480x480_sam_adam.pth.onnx")
assert onnx_path.exists()
model = onnx.load(onnx_path.as_posix())

onnx.checker.check_model(onnx_path.as_posix(), full_check=True)
print(onnx.helper.printable_graph(model.graph))
t = cv2.cvtColor(cv2.imread("../imgs/validation_epoch_0.png"), cv2.COLOR_BGR2RGB)


def extract_face(frame):
    image = cv2.resize(frame, (480, 480))
    tensor = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = np.transpose(tensor, (2, 0, 1))
    sess = rt.InferenceSession(onnx_path.as_posix())
    input_name = sess.get_inputs()[0].name
    try:
        bbxs = sess.run(None, {input_name: tensor})[0]
    except Exception as e:
        bbxs = []
    print(bbxs)
    for b in bbxs:
        if len(b) == 5:
            b = b[1:]
        if b[2] <= 15 or b[3] <= 15:
            width = 1
        else:
            width = 3
        bbx = [int(p) for p in convert_bbx_to_xyxy(b)]
        image = cv2.rectangle(image, pt1=(bbx[0], bbx[1]), pt2=(bbx[2], bbx[3]), thickness=width, color=(0, 0, 200))
    return image

# extract_face(t)
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
