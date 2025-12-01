import torch
import math
import numbers
import random
import numpy as np
import cv2

from PIL import Image, ImageOps, ImageEnhance


class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size  # (h, w)
        self.padding = padding

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        flow = sample.get('flow', None)
        forward = sample.get('forward', None)
        masks = sample.get('masks', None)
        gray=sample['gray']

        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)
            if flow is not None:
                flow = ImageOps.expand(flow, border=self.padding, fill=0)
            if forward is not None:
                forward = ImageOps.expand(Image.fromarray(forward), border=self.padding, fill=0)
                forward = np.array(forward)
            if masks is not None:
                masks = ImageOps.expand(Image.fromarray(masks), border=self.padding, fill=0)
                masks = np.array(masks)
            if gray is not None:
                gray = ImageOps.expand(gray, border=self.padding, fill=0)

        w, h = img.size
        th, tw = self.size

        if w == tw and h == th:
            return sample

        if w < tw or h < th:
            img = img.resize((tw, th), Image.BILINEAR)
            mask = mask.resize((tw, th), Image.NEAREST)
            if flow is not None:
                flow = flow.resize((tw, th), Image.BILINEAR)
            if forward is not None:
                forward = cv2.resize(forward, (tw, th), interpolation=cv2.INTER_NEAREST)
            if masks is not None:
                masks = cv2.resize(masks, (tw, th), interpolation=cv2.INTER_NEAREST)
            if gray is not None:
                gray = gray.resize((tw, th), Image.BILINEAR)
        else:
            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)
            img = img.crop((x1, y1, x1 + tw, y1 + th))
            mask = mask.crop((x1, y1, x1 + tw, y1 + th))
            if flow is not None:
                flow = flow.crop((x1, y1, x1 + tw, y1 + th))
            if forward is not None:
                forward = forward[y1:y1 + th, x1:x1 + tw]
            if masks is not None:
                masks = masks[y1:y1 + th, x1:x1 + tw]
            if gray is not None:
                gray = gray.crop((x1, y1, x1 + tw, y1 + th))
        sample['image'] = img
        sample['label'] = mask
        if flow is not None:
            sample['flow'] = flow
        if forward is not None:
            sample['forward'] = forward
        if masks is not None:
            sample['masks'] = masks
        if gray is not None:
            sample['gray'] = gray

        return sample


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        flow = sample['flow']
        assert img.size == mask.size
        assert img.size == flow.size
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        img = img.crop((x1, y1, x1 + tw, y1 + th))
        mask = mask.crop((x1, y1, x1 + tw, y1 + th))
        flow = flow.crop((x1, y1, x1 + tw, y1 + th))

        return {'image': img,
                'label': mask,
                'flow': flow}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        gray=sample['gray']
        depth = sample['depth']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            gray=gray.transpose(Image.FLIP_LEFT_RIGHT)
            mask = Image.fromarray(mask)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            mask = np.array(mask)
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)



        return {'image': img,
                'label': mask,
                'gray': gray,
                'depth': depth}


class Normalize(object):
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = np.array(sample['image']).astype(np.float32) / 255.0
        mask = sample['label'].astype(np.float32)

        img -= self.mean
        img /= self.std

        output = {
            'image': img,
            'label': mask,
        }

        if 'forward' in sample:
            output['forward'] = sample['forward'].astype(np.float32)
        if 'masks' in sample:
            output['masks'] = sample['masks'].astype(np.float32)
        if 'gray' in sample:
            output['gray'] = np.array(sample['gray']).astype(np.float32)/255.0

        return output


class ToTensor(object):
    def __call__(self, sample):
        img = np.array(sample['image']).astype(np.float32).transpose((2, 0, 1))
        mask = np.expand_dims(sample['label'].astype(np.float32), -1).transpose((2, 0, 1))
        mask[mask == 255] = 0

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()

        output = {'image': img, 'label': mask}

        if 'forward' in sample:
            forward = np.expand_dims(sample['forward'], axis=0)
            output['forward'] = torch.from_numpy(forward).float()

        if 'masks' in sample:
            masks = np.expand_dims(sample['masks'], axis=0)
            output['masks'] = torch.from_numpy(masks).float()
        if 'gray' in sample:
            gray = np.expand_dims(sample['gray'].astype(np.float32), -1).transpose((2, 0, 1))
            output['gray'] = torch.from_numpy(gray).float()

        return output

class FixedResize(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)

    def __call__(self, sample):
        img = sample['image']
        label = sample['label']
        forward = sample.get('forward', None)
        masks = sample.get('masks', None)
        gray=sample['gray']
        img = img.resize(self.size, Image.BILINEAR)
        label = cv2.resize(label, self.size, interpolation=cv2.INTER_NEAREST)

        if forward is not None:
            forward = cv2.resize(forward, self.size, interpolation=cv2.INTER_NEAREST)
        if masks is not None:
            masks = cv2.resize(masks, self.size, interpolation=cv2.INTER_NEAREST)
        if gray is not None:
            gray = gray.resize(self.size, Image.BILINEAR)
        output = {
            'image': img,
            'label': label,
        }

        if forward is not None:
            output['forward'] = forward
        if masks is not None:
            output['masks'] = masks
        if gray is not None:
            output['gray'] = gray

        return output


class Scale(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        assert img.size == mask.size
        assert img.size == flow.size

        w, h = img.size

        if (w >= h and w == self.size[1]) or (h >= w and h == self.size[0]):
            return {'image': img,
                    'label': mask,
                    'flow': flow}
        oh, ow = self.size
        img = img.resize((ow, oh), Image.BILINEAR)
        flow = flow.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        return {'image': img,
                'label': mask,
                'flow': flow}


class RandomSizedCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        flow = sample['flow']
        edge = sample['edge']
        assert img.size == mask.size
        assert img.size == flow.size
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.45, 1.0) * area
            aspect_ratio = random.uniform(0.5, 2)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                mask = mask.crop((x1, y1, x1 + w, y1 + h))
                flow = flow.crop((x1, y1, x1 + w, y1 + h))

                assert (img.size == (w, h))

                img = img.resize((self.size, self.size), Image.BILINEAR)
                mask = mask.resize((self.size, self.size), Image.NEAREST)
                flow = flow.resize((self.size, self.size), Image.BILINEAR)

                return {'image': img,
                        'label': mask,
                        'flow': flow}

        # Fallback
        scale = Scale(self.size)
        crop = CenterCrop(self.size)
        sample = crop(scale(sample))
        return sample

class RandomRotateOrthogonal(object):

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        flow = sample['flow']

        rotate_degree = random.randint(0, 3) * 90
        if rotate_degree > 0:
            img = img.rotate(rotate_degree, Image.BILINEAR)
            mask = mask.rotate(rotate_degree, Image.NEAREST)
            flow = flow.rotate(rotate_degree, Image.BILINEAR)

        return {'image': img,
                'label': mask,
                'flow': flow}


class RandomSized(object):
    def __init__(self, size):
        self.size = size
        self.scale = Scale(self.size)
        self.crop = RandomCrop(self.size)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        flow = sample['flow']
        assert img.size == mask.size
        assert img.size == flow.size

        w = int(random.uniform(0.8, 2.5) * img.size[0])
        h = int(random.uniform(0.8, 2.5) * img.size[1])

        img, mask = img.resize((w, h), Image.BILINEAR), mask.resize((w, h), Image.NEAREST)
        flow = flow.resize((w, h), Image.BILINEAR)

        sample = {'image': img, 'label': mask, 'flow': flow}

        return self.crop(self.scale(sample))


class RandomScale(object):
    def __init__(self, limit):
        self.limit = limit

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        flow = sample['flow']
        assert img.size == mask.size
        assert img.size == flow.size

        scale = random.uniform(self.limit[0], self.limit[1])
        w = int(scale * img.size[0])
        h = int(scale * img.size[1])

        img, mask = img.resize((w, h), Image.BILINEAR), mask.resize((w, h), Image.NEAREST)
        flow = flow.resize((w, h), Image.BILINEAR)

        return {'image': img, 'label': mask, 'flow': flow}


class RandomRotate(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        forward = sample.get('forward', None)
        masks = sample.get('masks', None)
        gray=sample['gray']
        if random.random() > 0.8:
            angle = np.random.randint(-15, 15)
            image = image.rotate(angle, Image.BICUBIC)
            label = Image.fromarray(label).rotate(angle, Image.NEAREST)
            label = np.array(label)

            if forward is not None:
                forward = Image.fromarray(forward).rotate(angle, Image.NEAREST)
                forward = np.array(forward)
            if masks is not None:
                masks = Image.fromarray(masks).rotate(angle, Image.NEAREST)
                masks = np.array(masks)
            if gray is not None:
                gray = gray.rotate(angle, Image.BICUBIC)

        output = {
            'image': image,
            'label': label,
        }
        if forward is not None:
            output['forward'] = forward
        if masks is not None:
            output['masks'] = masks
        if gray is not None:
            output['gray'] = gray

        return output


class colorEnhance(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        forward = sample['forward']
        masks = sample['masks']
        gray=sample['gray']
        bright_intensity = random.randint(5, 15) / 10.0
        image = ImageEnhance.Brightness(image).enhance(bright_intensity)
        contrast_intensity = random.randint(5, 15) / 10.0
        image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
        color_intensity = random.randint(0, 20) / 10.0
        image = ImageEnhance.Color(image).enhance(color_intensity)
        sharp_intensity = random.randint(0, 30) / 10.0
        image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
        
        return {'image': image,
                'label': label,
                'forward': forward,
                'masks': masks,
                'gray': gray}


class randomPeper(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        forward = sample.get('forward', None)
        masks = sample.get('masks', None)
        gray=sample['image']

        label = label.copy()  # 원본 유지
        noise_num = int(0.0015 * label.shape[0] * label.shape[1])
        for _ in range(noise_num):
            x = random.randint(0, label.shape[0] - 1)
            y = random.randint(0, label.shape[1] - 1)
            label[x, y] = random.choice([0, 1])

        output = {
            'image': image,
            'label': label
        }
        if forward is not None:
            output['forward'] = forward
        if masks is not None:
            output['masks'] = masks
        if gray is not None:
            output['gray'] = gray

        return output


class RandomFlip(object):
    def __call__(self, sample):
        img = sample['image']
        label = sample['label']
        forward = sample.get('forward', None)
        masks = sample.get('masks', None)
        gray=sample['gray']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            
            label = Image.fromarray(label).transpose(Image.FLIP_LEFT_RIGHT)
            label = np.array(label)

            if forward is not None:
                forward = np.fliplr(forward)
            if masks is not None:
                masks = np.fliplr(masks)
            if gray is not None:
                gray=gray.transpose(Image.FLIP_LEFT_RIGHT)
        output = {
            'image': img,
            'label': label,
        }

        if forward is not None:
            output['forward'] = forward
        if masks is not None:
            output['masks'] = masks
        if gray is not None:
            output['gray'] = gray

        return output
