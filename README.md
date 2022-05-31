## PyTorch Face Detection from Scratch

This project is a challenge to myself to be able to learn and implement face detection from scratch.
To do this I use the following Python libraries:
- [PyTorch](https://pytorch.org/)
- [OpenCV](https://opencv.org/)
- [PyTorch Lightning](https://www.pytorchlightning.ai/)
- [Albumentations](https://albumentations.ai/)

I used original [Yolo Paper](https://arxiv.org/abs/1506.02640) as a reference to implement the algorithms involved.

I experimented with different backbones and loss function hyperparameters (such as penalty weighting for incorrect predictions).
From my testing I concluded several points:
- The Yolo loss function performs poorly on very small objects close together (e.g. a crowd of faces).
- The Yolo loss function performs well on larger objects (e.g. a single face).
- The Yolo loss function is very sensitive to over or under tuning the hyperparameters.


The more interesting part I found about the project was trying to get the model as small as possible, while maintaining
the same accuracy. For this I tried 3 different architectures:
- A standard Resnet backbone, to get a good starting point.
- A Pool Resnet backbone, to reduce the computation time, without changing the parameters.
- A pretrained Mobilenet v3 backbone, to test if pretraining made any different.


I found that:
- The standard Resnet backbone performs well, but can be too slow when the number of bounding boxes is greater than 100.
- The Pool Resnet backbone performs equally well, but is much faster.
- The Mobilenet v3 backbone performs the same as the Pool Resnet, leading me to believe that none of the pretraining is
helping.


### Run the model
To run the medium PoolResnet Model:
1. Install the requirements
```bash
pip install -r requirements.txt
```
2. Run the model
```bash
python demo_scripts/demo_model_torch.py
```

### Train a model
To run the medium PoolResnet Model:
1. Install the requirements
```bash
pip install -r requirements.txt
```
2. Run the model
```bash
python demo_scripts/demo_model_torch.py
```