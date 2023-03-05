import torch.utils.data as data
import torchvision.transforms as transforms
import os
from PIL import Image
import random
import numpy as np


IMG_EXTENSIONS= [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def kt2012_loader(filepath):

    left_path = os.path.join(filepath, 'colored_0')
    right_path = os.path.join(filepath, 'colored_1')
    displ_path = os.path.join(filepath, 'disp_occ')

    total_name = [name for name in os.listdir(left_path) if name.find('_10') > -1]
    train_name = total_name[:160]
    val_name = total_name[160:]

    train_left = []
    train_right = []
    train_displ = []
    for name in train_name:
        train_left.append(os.path.join(left_path, name))
        train_right.append(os.path.join(right_path, name))
        train_displ.append(os.path.join(displ_path, name))

    val_left = []
    val_right = []
    val_displ = []
    for name in val_name:
        val_left.append(os.path.join(left_path, name))
        val_right.append(os.path.join(right_path, name))
        val_displ.append(os.path.join(displ_path, name))

    return train_left, train_right, train_displ, val_left, val_right, val_displ


def img_loader(path):
    return Image.open(path).convert('RGB')


def disparity_loader(path):
    return Image.open(path)


transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


class myDataset(data.Dataset):

    def __init__(self, left, right, left_disp, training, imgloader=img_loader, disploader=disparity_loader):
        self.left = left
        self.right = right
        self.left_disp = left_disp
        self.imgloader = imgloader
        self.disploader = disploader
        self.training = training

    def __getitem__(self, index):
        left = self.left[index]
        right = self.right[index]
        left_disp = self.left_disp[index]

        limg = self.imgloader(left)
        rimg = self.imgloader(right)
        ldisp = self.disploader(left_disp)

        if self.training:
            w, h = limg.size
            tw, th = 512, 256

            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            limg = limg.crop((x1, y1, x1 + tw, y1 + th))
            rimg = rimg.crop((x1, y1, x1 + tw, y1 + th))
            ldisp = np.ascontiguousarray(ldisp, dtype=np.float32)/256
            ldisp = ldisp[y1:y1 + th, x1:x1 + tw]

            limg = transform(limg)
            rimg = transform(rimg)

            return limg, rimg, ldisp

        else:
            w, h = limg.size

            limg = limg.crop((w-1232, h-368, w, h))
            rimg = rimg.crop((w-1232, h-368, w, h))
            ldisp = ldisp.crop((w-1232, h-368, w, h))
            ldisp = np.ascontiguousarray(ldisp, dtype=np.float32)/256

            limg = transform(limg)
            rimg = transform(rimg)

            return limg, rimg, ldisp

    def __len__(self):
        return len(self.left)