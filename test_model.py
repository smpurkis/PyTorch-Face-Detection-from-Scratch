from pathlib import Path

import torch
from pytorch_lightning import Trainer

from datasets.WIDERFace import WIDERFaceDataModule
from models import BaseModel, ModelMeta

if __name__ == '__main__':
    torch.random.manual_seed(0)
    Path("logs/out_resnet_custom_64_15x15_480x480_sam_adam.log").unlink(missing_ok=True)

    num_of_patches = 10
    input_shape = (480, 480)

    model = BaseModel(
        filters=64,
        input_shape=(3, *input_shape),
        num_of_patches=num_of_patches,
        num_of_residual_blocks=10
    ).cuda()

    model.summary()
    model_setup = ModelMeta(
        model=model,
        lr=1e-4
    )

    # checkpoint = torch.load("lightning_logs/custom_64_10x10_sam_adam/checkpoints/epoch=69-step=56279.ckpt")
    # model_setup.load_state_dict(checkpoint["state_dict"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(
        gpus=1 if device == torch.device("cuda") else 0,
        max_epochs=70,
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
    trainer.test(model=model_setup, test_dataloaders=dm.val_dataloader())
