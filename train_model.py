import torch
from pytorch_lightning import Trainer

from datasets.WIDERFace import WIDERFaceDataModule
from models import BaseModel, SetupModelTraining

if __name__ == '__main__':
    checkpoint = torch.load("lightning_logs/version_210/checkpoints/epoch=27-step=5627.ckpt")

    num_of_patches = 5
    input_shape = (320, 320)
    model = BaseModel(
        filters=64,
        input_shape=(3, *input_shape),
        num_of_patches=num_of_patches,
        num_of_residual_blocks=10
    ).cuda()
    model.summary()
    model_setup = SetupModelTraining(
        model=model,
        lr=1e-5
    )
    model_setup.load_state_dict(checkpoint["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(
        gpus=1 if device == torch.device("cuda") else 0,
        max_epochs=100,
        precision=16,
        progress_bar_refresh_rate=1,
        num_sanity_val_steps=0
    )
    dm = WIDERFaceDataModule(
        batch_size=8,
        input_shape=input_shape,
        num_of_patches=num_of_patches,
        shuffle=False
    )
    trainer.fit(model=model_setup, datamodule=dm)
