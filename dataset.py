import random

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from PIL import Image


def random_rotate(img, mask):
    angle = random.choice([30, -30, 60, -60, 90, -90])
    return TF.rotate(img, angle), TF.rotate(mask, angle)


class MyDataset(torch.utils.data.Dataset):

    def __init__(self, img_ids, is_train):
        self.img_id = img_ids
        self.is_train = is_train
        self.imgs = []
        self.labels = []

        self.transform_img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(0.5, 1)
        ])
        self.transform_label = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.NEAREST)
        ])

        for img_id in self.img_id:
            img_path = f'DRIVE/training/images/{img_id}_training.tif'
            self.imgs.append(img_path)
            label_path = f'DRIVE/training/1st_manual/{img_id}_manual1.gif'
            self.labels.append(label_path)

    def __getitem__(self, item):
        img = self.imgs[item]
        label = self.labels[item]

        img = Image.open(img)
        label = Image.open(label)

        img = self.transform_img(img).float()
        label = self.transform_label(label).float()

        # train mode下加入数据增强
        if self.is_train is True:
            # 0.5的概率选择是否旋转
            if random.randint(0, 1):
                img, label = random_rotate(img, label)

        return img, label, self.img_id[item]

    def __len__(self):
        return len(self.imgs)