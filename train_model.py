import torch

from pytorch_lightning import Trainer

if __name__ == '__main__':
    from models import BaseModel, SetupModelTraining
    from datasets.WIDERFace import WIDERFaceDataModule

    model = BaseModel(
        filters=64,
        input_shape=(3, 320, 320),
        num_of_patches=20,
        num_of_residual_blocks=5
    )
    model_setup = SetupModelTraining(
        model=model,
        lr=1e-4
    )
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
        shuffle=False
    )
    trainer.fit(model=model_setup, datamodule=dm)
