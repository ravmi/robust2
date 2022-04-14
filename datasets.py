import abc
import functools
import random
from typing import List

import cv2
import numpy as np
import torch
import torchvision
import tqdm
from pytorch_lightning.utilities import data
from torch.utils.data import Dataset
import pytorch_lightning as pl
import matplotlib.pyplot as plt


class AbstractDenseGraspDataset(abc.ABC, Dataset):
    def __init__(self, directory="/scidata/grip/dense"):
        self.directory = directory

    def get_image_by_index(self, index):
        index += 100   # numbering starts at 100
        image = cv2.imread(f'{self.directory}/one_side_try{index}/original.png')
        target = cv2.imread(f'{self.directory}/one_side_try{index}/result-0.png')
        target = (target[:, :, 0] != 84).astype(np.float32)
        return image, target

    @abc.abstractmethod
    def __getitem__(self, item):
        pass

    @abc.abstractmethod
    def __len__(self):
        pass


class DenseGraspDatasetsConcat(AbstractDenseGraspDataset):
    def __init__(self, datasets: List[AbstractDenseGraspDataset]):
        super().__init__(None)
        assert len(datasets) == 2, "Only two datasets are supported"
        assert datasets[0].directory == datasets[1].directory
        self.datasets = datasets

    def get_image_by_index(self, index):
        assert False

    def __getitem__(self, index):
        l1, l2 = len(self.datasets[0]), len(self.datasets[1])

        if index > l1:
            return self.datasets[0].__getitem__(index - l1)

        return self.datasets[1].__getitem__(index)

    def __len__(self):
        return sum(len(d) for d in self.datasets)


class DenseGraspDataset(AbstractDenseGraspDataset):
    def __init__(self, size, offset=0, directory="/scidata/grip/dense", cutout=None):
        super().__init__(directory)
        self.transforms = torchvision.transforms.ToTensor()
        self.size = size
        self.offset = offset
        self.cutout = cutout

    def __getitem__(self, index):
        image, output = self.get_image_by_index(index + self.offset)
        mask = np.ones_like(output)
        if self.cutout:
            self.cutout.cut(image, mask)
        img_tensor = self.transforms(image)
        out_tensor = self.transforms(output)
        mask = self.transforms(mask)

        return img_tensor, out_tensor, mask

    def __len__(self):
        return self.size


class DenseGraspDataModule(pl.LightningDataModule):
    def __init__(self, train_dataset: AbstractDenseGraspDataset, test_dataset: AbstractDenseGraspDataset, batch_size):
        super().__init__()
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size

    def train_dataloader(self):
        return data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            drop_last=True,
            pin_memory=True,
            num_workers=4,
            shuffle=True,
        )

    def val_dataloader(self):
        return data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            drop_last=True,
            pin_memory=True,
            num_workers=4,
            shuffle=False,
        )


class Cutout:
    def __init__(self, patch_count, patch_size):
        self.patch_count = patch_count
        self.patch_size = patch_size

    def cut(self, image, mask):
        for i in range(self.patch_count):
            x = random.randint(0, 224 - self.patch_size - 1)
            y = random.randint(0, 224 - self.patch_size - 1)
            image[x:x + self.patch_size, y:y + self.patch_size] = [0, 0, 0]
            mask[x:x + self.patch_size, y:y + self.patch_size] = 0


class SoftDataset(AbstractDenseGraspDataset):
    def __init__(self, teacher, size, weight, directory="/scidata/grip/dense"):
        super().__init__(directory)
        self.teacher = teacher
        self.transforms = torchvision.transforms.ToTensor()
        self.size = size
        self.weight = weight

        for i in tqdm.tqdm(range(size), desc="Evaluating dataset on teachers"):
            self.__getitem__(i)

    @functools.lru_cache()
    def __getitem__(self, index):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        image, _ = self.get_image_by_index(index)
        img_tensor = self.transforms(image)
        img_tensor = img_tensor.to(device)

        results = []
        self.teacher = self.teacher.to(device)
        self.teacher.eval()
        with torch.no_grad():
            results.append(torch.sigmoid(self.teacher(img_tensor.unsqueeze(0))))

        out_tensor = torch.stack(results).mean(dim=0)[0, :, :, :].to("cpu")
        return img_tensor.to("cpu"), out_tensor, self.weight * torch.ones_like(out_tensor)

    def __len__(self):
        return self.size


if __name__ == "__main__":
    def show_dataset(ds):
        plt.figure(figsize=(10, 6))
        for i in range(len(ds)):
            sample, target, mask = ds[i]
            ax = plt.subplot(3, 4, i + 1)
            ax.set_title('Sample #{}'.format(i))
            plt.imshow(sample.permute(1, 2, 0).cpu())
            plt.subplot(3, 4, i + 5)
            plt.imshow(target[0, :, :].cpu(), vmin=0, vmax=1)
            plt.subplot(3, 4, i + 9)
            plt.imshow(mask[0, :, :].cpu())
            if i == 3:
                plt.show()
                break

    train_ds = DenseGraspDataset(5, directory="grip/dense", cutout=Cutout(20, 20))
    show_dataset(train_ds)
