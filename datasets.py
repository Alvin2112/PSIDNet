import glob
import os.path

import torch
import random
import torchvision.transforms as tfs
from torchvision.transforms import functional as FF
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset, DataLoader


class PairedImgDataset(Dataset):
    def __init__(self, data_source, mode, crop=256, random_resize=None):
        if not mode in ['train', 'val', 'test']:
            raise Exception('The mode should be "train", "val" or "test".')

        self.random_resize = random_resize
        self.crop = crop
        self.mode = mode
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        self.dataset = 'RESIDE_indoor'

        if self.mode == 'train':
            dir_root = '../data'
            self.haze_left = sorted(glob.glob(dir_root  + '/' + 'train'+ '/left' + '/haze' + '/*.*'))
            self.gt_left = sorted(glob.glob(dir_root  + '/' + 'train' + '/left'+ '/clear' + '/*.*'))

        if self.mode == 'val':
            dir_root = '../data'
            self.haze_left = sorted(glob.glob(dir_root + '/' + 'val' +'/left' + '/haze' + '/*.*'))
            self.gt_left = sorted(glob.glob(dir_root  + '/' + 'val' +'/left' + '/clear' + '/*.*'))

        if self.mode == 'test':
            dir_root = '../data'
            self.haze_left = sorted(glob.glob(dir_root  + '/' + 'test' +'/left'  + '/haze' + '/*.*'))
            self.gt_left = sorted(glob.glob(dir_root  + '/' + 'test' + '/left' + '/clear' + '/*.*'))
                


    def __getitem__(self, index):

        if self.mode == 'train':
            image_name = self.haze_left[index % len(self.haze_left)].split('/')
            #print(image_name)
            haze_left_dir = os.path.join('../data/train/left/haze/',image_name[-1])
            haze_right_dir = os.path.join('../data/train/right/haze/',image_name[-1])
            gt_left_dir = os.path.join('../data/train/left/clear/',image_name[-1])
            gt_right_dir = os.path.join('../data/train/right/clear/',image_name[-1])
        
        if self.mode == 'val':
            image_name = self.haze_left[index % len(self.haze_left)].split('/')
            # print(image_name)
            
            haze_left_dir = os.path.join('../data/val/left/haze/',image_name[-1])
            haze_right_dir = os.path.join('../data/val/right/haze/',image_name[-1])
            gt_left_dir = os.path.join('../data/val/left/clear/',image_name[-1])
            gt_right_dir = os.path.join('../data/val/right/clear/',image_name[-1])

        if self.mode == 'test':
            image_name = self.haze_left[index % len(self.haze_left)].split('/')
            # print(image_name)

            haze_left_dir = os.path.join('../data/test/left/haze/',image_name[-1])
            haze_right_dir = os.path.join('../data/test/right/haze/',image_name[-1])
            gt_left_dir = os.path.join('../data/test/left/clear/',image_name[-1])
            gt_right_dir = os.path.join('../data/test/right/clear/',image_name[-1])           

        #print("haze_img_dir:",haze_img_dir)
        #print("clear_img_dir:",clear_img_dir)

        haze_left = Image.open(haze_left_dir).convert('RGB')
        haze_right = Image.open(haze_right_dir).convert('RGB')
        gt_left = Image.open(gt_left_dir).convert('RGB')
        gt_right = Image.open(gt_right_dir).convert('RGB')

        haze_left = self.transform( haze_left)
        haze_right = self.transform( haze_right)
        gt_left = self.transform( gt_left)
        gt_right = self.transform( gt_right)

        if self.mode == 'train':
            if self.random_resize is not None:
                # random resize
                scale_factor = random.uniform(self.crop / self.random_resize, 1.)
                img = F.interpolate(img.unsqueeze(0), scale_factor=scale_factor, align_corners=False, mode='bilinear',
                                    recompute_scale_factor=False).squeeze(0)
                gt = F.interpolate(gt.unsqueeze(0), scale_factor=scale_factor, align_corners=False, mode='bilinear',
                                   recompute_scale_factor=False).squeeze(0)
            # crop
            h, w = haze_left.size(1), haze_left.size(2)
            offset_h = random.randint(0, max(0, h - self.crop - 1))
            offset_w = random.randint(0, max(0, w - self.crop - 1))

            haze_left = haze_left[:, offset_h:offset_h + self.crop, offset_w:offset_w + self.crop]
            haze_right = haze_right[:, offset_h:offset_h + self.crop, offset_w:offset_w + self.crop]
            gt_left = gt_left[:, offset_h:offset_h + self.crop, offset_w:offset_w + self.crop]
            gt_right = gt_right[:, offset_h:offset_h + self.crop, offset_w:offset_w + self.crop]


        return haze_left,haze_right,gt_left,gt_right

    def __len__(self):
        return max(len(self.haze_left), len(self.gt_left))



class SingleImgDataset(Dataset):
    def __init__(self, data_source):
        
        self.img_paths = sorted(glob.glob(data_source + '/' + 'val' + '/input' + '/*.*'))

    def __getitem__(self, index):
        
        path = self.img_paths[index % len(self.img_paths)]
        
        img = np.load(path)
        
        img = self.tone_map(img)
        
        img = self.to_tensor(img)
        
        return img, path

    def __len__(self):
        return len(self.img_paths)
    
    def tone_map(self, x):
        return x / (x + 0.25)
    
    def to_tensor(self, x):
        """Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)."""
        x = np.transpose(x, (2, 0, 1))
        x  = torch.from_numpy(x).float()
        return x
