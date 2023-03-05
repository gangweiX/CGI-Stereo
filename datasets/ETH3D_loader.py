import os
from PIL import Image
from datasets import readpfm as rp
# import dataloader.preprocess
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random

IMG_EXTENSIONS= [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


# filepath = '/media/data/dataset/ETH3D/'
def et_loader(filepath):

    left_img = []
    right_img = []
    disp_gt = []
    occ_mask = []

    img_path = os.path.join(filepath, 'two_view_training')
    gt_path = os.path.join(filepath, 'two_view_training_gt')

    for c in os.listdir(img_path):
        img_cpath = os.path.join(img_path, c)
        gt_cpath = os.path.join(gt_path, c)

        left_img.append(os.path.join(img_cpath, 'im0.png'))
        right_img.append(os.path.join(img_cpath, 'im1.png'))
        disp_gt.append(os.path.join(gt_cpath, 'disp0GT.pfm'))
        occ_mask.append(os.path.join(gt_cpath, 'mask0nocc.png'))

    return left_img, right_img, disp_gt, occ_mask,


def img_loader(path):
    return Image.open(path).convert('RGB')


def disparity_loader(path):
    return rp.readPFM(path)


class myDataset(data.Dataset):

    def __init__(self, left, right, disp_gt, occ_mask, training, imgloader=img_loader, dploader = disparity_loader):
        self.left = left
        self.right = right
        self.disp_gt = disp_gt
        self.occ_mask = occ_mask
        self.imgloader = imgloader
        self.dploader = dploader
        self.training = training
        self.img_transorm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __getitem__(self, index):
        left = self.left[index]
        right = self.right[index]
        disp_L = self.disp_L[index]
        disp_R = self.disp_R[index]

        left_img = self.imgloader(left)
        right_img = self.imgloader(right)
        dataL, _ = self.dploader(disp_L)
        dataL = np.ascontiguousarray(dataL, dtype=np.float32)
        dataR, _ = self.dploader(disp_R)
        dataR = np.ascontiguousarray(dataR, dtype=np.float32)

        if self.training:
            w, h = left_img.size
            tw, th = 512, 256
            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            left_img = left_img.crop((x1, y1, x1+tw, y1+th))
            right_img = right_img.crop((x1, y1, x1+tw, y1+th))
            dataL = dataL[y1:y1+th, x1:x1+tw]
            dataR = dataR[y1:y1+th, x1:x1+tw]

            left_img = self.img_transorm(left_img)
            right_img = self.img_transorm(right_img)

            return left_img, right_img, dataL, dataR

        else:
            w, h = left_img.size
            left_img = left_img.crop((w-960, h-544, w, h))
            right_img = right_img.crop((w-960, h-544, w, h))

            left_img = self.img_transorm(left_img)
            right_img = self.img_transorm(right_img)

            dataL = Image.fromarray(dataL).crop((w-960, h-544, w, h))
            dataL = np.ascontiguousarray(dataL)
            dataR = Image.fromarray(dataR).crop((w-960, h-544, w, h))
            dataR = np.ascontiguousarray(dataR)

            return left_img, right_img, dataL, dataR

    def __len__(self):
        return len(self.left)





