import os
from PIL import Image
import torch
from torch.utils import data
import numpy as np


def randomCrop(image, label, forward, back):
    border = 30
    image_width, image_height = image.size

    # crop 사이즈 랜덤 설정
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)

    # crop 박스 좌표 계산
    left = (image_width - crop_win_width) >> 1
    upper = (image_height - crop_win_height) >> 1
    right = (image_width + crop_win_width) >> 1
    lower = (image_height + crop_win_height) >> 1
    box = (left, upper, right, lower)

    # 모든 입력을 PIL 이미지로 변환 후 crop
    label = Image.fromarray(label)
    forward = Image.fromarray(forward)
    back = Image.fromarray(back)

    return (
        image.crop(box),
        np.array(label.crop(box)),
        np.array(forward.crop(box)),
        np.array(back.crop(box))
    )


class Dataset(data.Dataset):
    def __init__(self, root_dir, mode='train', transform=None, return_size=True):
        """
        root_dir : 예) 'D:/remote/S-EOR'
        mode     : 'train' 또는 'test'
        """
        self.return_size = return_size
        self.mode = mode
        self.transform = transform

        img_dir = os.path.join('/mnt/d/Polyp/',root_dir, mode, 'image')
        gt_dir  = os.path.join('/mnt/d/Polyp/',root_dir, mode, 'pseudo_mask')
        scribble_dir=os.path.join('/mnt/d/Polyp/',root_dir, mode, 'scribble')

        img_files = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

        self.datas_id = []
        for fname in img_files:
            img_path = os.path.join(img_dir, fname)
            gt_name  = os.path.splitext(fname)[0] + '.png'

            gt_path  = os.path.join(gt_dir, gt_name)
            scribble_path=os.path.join(scribble_dir,gt_name)
            if not os.path.exists(gt_path):
                continue

            self.datas_id.append({
                'img_path': img_path,
                'gt_path': gt_path,
                'scribble_path': scribble_path,
                'name': fname
            })

    def __getitem__(self, idx):
        meta = self.datas_id[idx]

        image = Image.open(meta['img_path']).convert('RGB')
        label = Image.open(meta['gt_path']).convert('L')
        scribble = Image.open(meta['scribble_path']).convert('L')

        label = np.array(label, dtype=np.float32)
        if label.max() > 0:
            label = label / 255.0

        scribble_np = np.array(scribble, dtype=np.uint8)
        forward = (scribble_np == 1).astype(np.float32)
        back = (scribble_np == 2).astype(np.float32)

        if self.mode == 'train':
            image, label, forward, back = randomCrop(image, label, forward, back)

        h, w = image.size[1], image.size[0]
        sample = {
            'image': image,
            'label': label,
            'forward': forward,
            'back': back
        }

        if self.transform:
            sample = self.transform(sample)

        if self.return_size:
            sample['size'] = torch.tensor([h, w])

        sample['name'] = meta['name']
        return sample
    def __len__(self):
        return len(self.datas_id)