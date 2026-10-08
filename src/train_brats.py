import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from monai.networks.nets import SwinUNETR
from monai.losses import DiceCELoss

from data_loader import get_brats_dataloaders

torch.set_float32_matmul_precision('high')


class BraTSModel(pl.LightningModule):
    """
    Swin-UNETR: a transformer-based encoder (Swin Transformer, using
    shifted-window self-attention) combined with a U-Net-style decoder and
    skip connections. Chosen over a plain CNN-based U-Net because gliomas
    grow in irregular, spread-out patterns, and Swin's windowed attention
    can relate distant regions of the brain to each other more effectively
    than a purely local convolutional receptive field.
    """
    def __init__(self):
        super().__init__()

        self.model = SwinUNETR(
            in_channels=4,     # T1, T1ce, T2, FLAIR stacked as 4 channels
            out_channels=3,    # Tumor Core, Whole Tumor, Enhancing Tumor
            feature_size=48,
            use_checkpoint=True,
            norm_name="instance",
            spatial_dims=3,
        )

        # DiceCELoss combines Dice Loss (rewards volume overlap) with
        # Cross-Entropy (rewards correct per-voxel classification) -- this
        # combination handles the severe class imbalance between tumor and
        # non-tumor voxels better than either loss alone.
        self.loss_function = DiceCELoss(to_onehot_y=False, sigmoid=True, squared_pred=True)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]

        # Defensive shape guard: MONAI's transform pipeline can occasionally
        # produce tensors with an unexpected extra dimension depending on
        # transform ordering -- this guards against that rather than
        # crashing training.
        if len(labels.shape) > 5:
            labels = labels[:, :, -96:, -96:, -96:]
            while len(labels.shape) > 5:
                labels = labels.squeeze(2)

        if len(images.shape) > 5:
            images = images[:, :, -96:, -96:, -96:]
            while len(images.shape) > 5:
                images = images.squeeze(2)

        outputs = self.forward(images)
        loss = self.loss_function(outputs, labels)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-5)
        return optimizer


if __name__ == "__main__":
    train_loader = get_brats_dataloaders(data_dir="./data", cache_dir="./cache", batch_size=1)

    model = BraTSModel()

    checkpoint_callback = ModelCheckpoint(
        dirpath="./checkpoints",
        filename="brats-best",
        save_last=True,
        monitor="train_loss",
        mode="min",
    )

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        max_epochs=100,
        callbacks=[checkpoint_callback],
        log_every_n_steps=10,
    )

    # Resume from the last checkpoint if one exists, otherwise start fresh.
    # (The original script always pointed at "last.ckpt", which would crash
    # on a first-ever run before any checkpoint exists -- this checks first.)
    resume_ckpt = "./checkpoints/last.ckpt"
    ckpt_path = resume_ckpt if os.path.exists(resume_ckpt) else None

    print("Starting training run...")
    trainer.fit(model, train_dataloaders=train_loader, ckpt_path=ckpt_path)
