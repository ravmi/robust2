from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from clearml import Task
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch import optim
from datasets import DenseGraspDataset, DenseGraspDataModule, DenseGraspDatasetRobust, DenseGraspDatasetRobustAugmented
from pgd import get_adv_examples
from steps import LinfStep, L2Step
import torchvision.transforms as T



def replace_best(loss, bloss, x, bx):
    if bloss is None:
        bx = x.clone().detach()
        bloss = loss.clone().detach()
    else:
        replace = bloss < loss
        bx[replace] = x[replace].clone().detach()
        bloss[replace] = loss[replace]

    return bloss, bx
















class SupervisedDenseGrasp(pl.LightningModule):
    def __init__(self, batch_size, lr, weight_decay, dropout, max_epochs, robust, iterations, eps, step_size, step_type, positive_loss_scaling=1.):
        super().__init__()
        self.save_hyperparameters()
        self.eps = eps
        self.step_size = step_size#0.01
        self.iterations = iterations
        self.robust=robust
        self.step_type=step_type


        self.img_log_step = 0
        self.bidx = 0

        self.log_iter  = 0
        #self.val_it = 0


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

    def step(self, batch, mode="train", batch_idx=0):
        img, labels, mask = batch
        assert img.shape == (self.hparams.batch_size, 3, 224, 224)
        assert labels.shape == (self.hparams.batch_size, 1, 224, 224)
        assert mask.shape == (self.hparams.batch_size, 1, 224, 224)

        outputs = self.forward(img)
        discretized_outputs = torch.as_tensor(torch.sigmoid(outputs) > 0.5, dtype=torch.long)

        #loss = torch.nn.BCEWithLogitsLoss(
        #    weight=((labels == 0) + (labels == 1) * self.hparams.positive_loss_scaling) * mask)(outputs, labels)
        pos_weight = torch.ones(224, 224, device=outputs.device) * 62.
        loss = torch.BCEWithLogitsLoss(pos_weight=pos_weight)

        for top_n in [None, 1, 5, 10, 20]:
            self.logging(mode, batch, outputs, loss, top_n)

        predimg = T.ToPILImage()(discretized_outputs[0] * 1.0)
        pimg = T.ToPILImage()(img[0])
        plabels = T.ToPILImage()(labels[0])

        logger.report_image(mode, "image_prediction", iteration=self.log_iter, image=predimg)
        if mode == "val":
            logger.report_image(mode, "img", iteration=self.log_iter, image=pimg)
        logger.report_image(mode, "labels", iteration=self.log_iter, image=plabels)

        self.log_iter += 1

        return loss

    def training_step(self, batch, batch_idx):
        # finding counterexamples here
        img, labels, mask = batch

        pimg = T.ToPILImage()(img[0])
        plabels = T.ToPILImage()(labels[0])
        logger.report_image("train", "img", iteration=self.log_iter, image=pimg)

        if self.robust:
            self.eval()

            #img, labels, mask = batch
            orig_input = img.clone().detach()
            if self.step_type == "inf":
                step = LinfStep(eps=self.eps, orig_input=orig_input, step_size=self.step_size)
            elif self.step_type == "l2":
                step = L2Step(eps=self.eps, orig_input=orig_input, step_size=self.step_size)
            x_adv = get_adv_examples(self.model, img, labels, step, self.iterations, random_start=False)
            batch = (x_adv, labels, mask)

            pdifimg = (img[0] - x_adv[0]).abs()
            self.log("mindif", pdifimg.min(), on_step=True, on_epoch=False)
            self.log("maxdif", pdifimg.max(), on_step=True, on_epoch=False)
            pimg = T.ToPILImage()(x_adv[0])
            pdifimg = T.ToPILImage()(pdifimg)
            logger.report_image("train", "image_adv", iteration=self.log_iter, image=pimg)
            logger.report_image("train", "pdifimg", iteration=self.log_iter, image=pdifimg)

            self.train()
        #logger.report_image("image", "image_lab", iteration=batch_idx, image=plabels)
        #img_log_step += 1
        self.bidx = batch_idx
        return self.step(batch, mode="train", batch_idx=batch_idx)

    def validation_step(self, batch, batch_idx):
        self.bidx = batch_idx
        return self.step(batch, mode="val", batch_idx=batch_idx)


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
    test_on_augmented = True#False#True
    robust = True#False#True
    iterations = 30
    #eps = 0.05
    eps = 50.
    step_size = 0.01
    step_type="l2"

    task_name = f"augmented: {test_on_augmented}, robust: {robust}, steps: {iterations}, eps: {eps} step_size: {step_size}, step_type: {step_type}"

    ###


    if test_on_augmented:
        train, test = DenseGraspDatasetRobust(450, directory="robust_dense"), DenseGraspDatasetRobustAugmented(50 * 10, 4500, directory="robust_dense")
        datamodule = DenseGraspDataModule(train, test, batch_size=8)
        task = Task.init(project_name="robustness", task_name=task_name, reuse_last_task_id=False)
    else:
        train, test = DenseGraspDatasetRobust(450, directory="robust_dense"), DenseGraspDatasetRobust(50, 450, directory="robust_dense")
        datamodule = DenseGraspDataModule(train, test, batch_size=8)
        task = Task.init(project_name="robustness", task_name=task_name, reuse_last_task_id=False)
    logger = task.get_logger()
    img_log_step = 0

    train_supervised(
        datamodule,
        model=None,
        batch_size=datamodule.batch_size,
        lr=1e-3,
        dropout=0.01,
        weight_decay=0.01,
        max_epochs=90,
        robust=robust,
        iterations=iterations,
        eps=eps,
        step_size=step_size,
        step_type=step_type
    )
