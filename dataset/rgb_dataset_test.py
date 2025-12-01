import os
from PIL import Image
import torch
from torch.utils import data
import numpy as np


def randomCrop(image, label):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    label = Image.fromarray(label)
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), np.array(label.crop(random_region))


class Dataset(data.Dataset):
    def __init__(self, root_dir, mode='train', transform=None, return_size=True):
        """
        root_dir : 예) 'D:/remote/S-EOR'
        mode     : 'train' 또는 'test'
        """
        self.return_size = return_size
        self.mode = mode
        self.transform = transform

        img_dir = os.path.join('/mnt/d/Camouflage/',root_dir, mode, 'image')
        gt_dir  = os.path.join('/mnt/d/Camouflage/',root_dir, mode, 'gt')

        img_files = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

        self.datas_id = []
        for fname in img_files:
            img_path = os.path.join(img_dir, fname)
            gt_name  = os.path.splitext(fname)[0] + '.png'
            gt_path  = os.path.join(gt_dir, gt_name)

            if not os.path.exists(gt_path):
                continue

            self.datas_id.append({
                'img_path': img_path,
                'gt_path': gt_path,
                'name': fname
            })

    def __getitem__(self, idx):
        meta = self.datas_id[idx]

        image = Image.open(meta['img_path']).convert('RGB')
        label = Image.open(meta['gt_path']).convert('L')
        label = np.array(label, dtype=np.float32)

        if label.max() > 0:
            label = label / 255.0

        h, w = image.size[1], image.size[0]
        sample = {'image': image, 'label': label}

        if self.mode == 'train':
            sample['image'], sample['label'] = randomCrop(sample['image'], sample['label'])

        if self.transform:
            sample = self.transform(sample)

        if self.return_size:
            sample['size'] = torch.tensor([h, w])
        
        sample['name'] = meta['name']
        
        return sample

    def __len__(self):
        return len(self.datas_id)