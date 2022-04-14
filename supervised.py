from typing import Optional
print('d')

import numpy as np
print('d')
import pytorch_lightning as pl
print('d')
import torch
import torchvision
print('d')
from clearml import Task
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch import optim

from datasets import DenseGraspDataset, DenseGraspDataModule, DenseGraspDatasetRobust, DenseGraspDatasetRobustAugmented


class SupervisedDenseGrasp(pl.LightningModule):
    def __init__(self, batch_size, lr, weight_decay, dropout, max_epochs, positive_loss_scaling=1.):
        super().__init__()
        self.save_hyperparameters()

        model = torchvision.models.segmentation.fcn_resnet50(pretrained=False)
        model.classifier[3] = torch.nn.Dropout(p=dropout)
        model.classifier[4] = torch.nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))
        self.model = model

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return optimizer

    def logging(self, mode, batch, outputs, loss, top_n: Optional[int] = None):
        img, labels, mask = batch
        labels = labels.reshape((labels.shape[0], -1))
        outputs = outputs.reshape((outputs.shape[0], -1))

        if top_n is not None:
            idxs = torch.argsort(outputs, dim=1)
            labels = labels[:, idxs[:, -top_n:]]
            outputs = outputs[:, idxs[:, -top_n:]]

        discretized_outputs = torch.as_tensor(torch.sigmoid(outputs) > 0.5, dtype=torch.long)

        def naming(label, top_n=top_n):
            return f"{mode}_{label}" if top_n is None else f"{mode}_top{top_n}_{label}"

        self.log(naming("loss", None), loss.mean(), on_step=True, on_epoch=True)
        self.log(naming("size", None), len(self.trainer.train_dataloader.dataset) if mode == "train" else len(
            self.trainer.val_dataloaders[0].dataset))
        self.log(naming("acc"), (torch.sum(discretized_outputs == labels) / np.prod(labels.shape)).item(), on_step=True,
                 on_epoch=True)
        if torch.sum(labels == 1).item() > 0:
            self.log(naming("recall"), (torch.sum(torch.logical_and(discretized_outputs == 1, labels == 1)) / torch.sum(
                labels == 1)).item(), on_step=True, on_epoch=True)
        self.log(naming("labels"), torch.sum(labels == 1).item(), on_step=True, on_epoch=True)
        self.log(naming("detections"), torch.sum(discretized_outputs == 1).item(), on_step=True, on_epoch=True)
        if torch.sum(discretized_outputs == 1).item() > 0:
            self.log(naming("precision"),
                     (torch.sum(torch.logical_and(discretized_outputs == 1, labels == 1)) /
                      torch.sum(discretized_outputs == 1)).item(), on_step=True, on_epoch=True)

    def forward(self, img):
        outputs = self.model(img)['out']
        return outputs

    def step(self, batch, mode="train"):
        img, labels, mask = batch
        assert img.shape == (self.hparams.batch_size, 3, 224, 224)
        assert labels.shape == (self.hparams.batch_size, 1, 224, 224)
        assert mask.shape == (self.hparams.batch_size, 1, 224, 224)

        outputs = self.forward(img)
        loss = torch.nn.BCEWithLogitsLoss(
            weight=((labels == 0) + (labels == 1) * self.hparams.positive_loss_scaling) * mask)(outputs, labels)

        for top_n in [None, 1, 5, 10, 20]:
            self.logging(mode, batch, outputs, loss, top_n)

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, mode="val")


def train_supervised(datamodule, model=None, **kwargs):
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    def train(**kwargs):
        trainer = pl.Trainer(
            default_root_dir="models/",
            gpus=1 if str(device) == "cuda:0" else 0,
            max_epochs=kwargs["max_epochs"],
            callbacks=[
                ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_loss"),
                LearningRateMonitor("step"),
            ],
            progress_bar_refresh_rate=1,
            log_every_n_steps=1,
        )

        nonlocal model
        model = model if model else SupervisedDenseGrasp(**kwargs)
        trainer.fit(model, datamodule)
        return model

    return train(**kwargs)


if __name__ == "__main__":
    #change this
    test_on_augmented = True

    ###


    if test_on_augmented:
        train, test = DenseGraspDatasetRobust(450, directory="robust_dense"), DenseGraspDatasetRobustAugmented(50 * 10, 4500, directory="robust_dense")
    else:
        train, test = DenseGraspDatasetRobust(450, directory="robust_dense"), DenseGraspDatasetRobust(50, 450, directory="robust_dense")

    datamodule = DenseGraspDataModule(train, test, batch_size=8)
    task = Task.init(project_name="robustness", task_name=f"dense grasping supervised with tests on augmented data", reuse_last_task_id=False)
    train_supervised(
        datamodule,
        model=None,
        batch_size=datamodule.batch_size,
        lr=1e-3,
        dropout=0.01,
        weight_decay=0.01,
        max_epochs=50,
        positive_loss_scaling=5.,
    )