import torch
from pytorch_lightning import Trainer

from datasets.WIDERFace import WIDERFaceDataModule
from models import BaseModel, SetupModelTraining

if __name__ == '__main__':
    checkpoint = torch.load("/home/sam/PycharmProjects/python/PyTorch-Face-Detection-from-Scratch/lightning_logs/version_2/checkpoints/epoch=37-step=61179.ckpt")
    model = BaseModel(
        filters=64,
        input_shape=(3, 320, 320),
        num_of_patches=20,
        num_of_residual_blocks=5
    )
    model_setup = SetupModelTraining(
        model=model,
        lr=1e-5
    )
    model_setup.load_state_dict(checkpoint["state_dict"])
    i = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # trainer = Trainer(
    #     gpus=1 if device == torch.device("cuda") else 0,
    #     max_epochs=100,
    #     precision=16,
    #     progress_bar_refresh_rate=1,
    #     num_sanity_val_steps=0
    # )
    dm = WIDERFaceDataModule(
        batch_size=8,
        shuffle=False
    )
    dm.setup()
    out = model_setup(dm.val_dataset[0][0].reshape(-1, *dm.val_dataset[0][0].shape))
    print(dm.val_dataset[0])
    # trainer.fit(model=model_setup, datamodule=dm)
