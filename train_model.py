from pathlib import Path

import torch
from pytorch_lightning import Trainer

from datasets.WIDERFace import WIDERFaceDataModule
from models import ModelMeta
from models.PoolResnet import PoolResnet
from models.Resnet import Resnet

if __name__ == '__main__':
    torch.random.manual_seed(0)
    Path("logs/out_resnet_custom_64_15x15_480x480_sam_adam.log").unlink(missing_ok=True)

    num_of_patches = nop = 15
    input_shape = (480, 480)
    filters = 64

    model = Resnet(
        filters=filters,
        input_shape=(3, *input_shape),
        num_of_patches=num_of_patches,
        num_of_residual_blocks=10
    ).cuda()

    model.summary()
    model_setup = ModelMeta(
        model=model,
        lr=1e-5
    )

    checkpoint = torch.load("lightning_logs/custom_resnet_64_15x15_480x480_sam_adam/checkpoints/epoch=52-step=42611.ckpt")
    model_setup.load_state_dict(checkpoint["state_dict"])

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
    trainer.fit(model=model_setup, datamodule=dm)
    model_setup.to_torchscript(f"./saved_models/custom_resnet_{filters}_{nop}x{nop}_{input_shape[0]}x{input_shape[1]}_sam_adam.pth")
    # trainer.test(model=model_setup, test_dataloaders=dm.test_dataloader())
