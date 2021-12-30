import os, sys
import numpy as np
import cv2
from PIL import Image, ImageDraw
import json

import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms

def json_parse(image_dir, json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    file_name_list = []
    anno = []
    for image_name in data:
        file_name_list.append(os.path.join(image_dir, image_name))
        anno.append(data[image_name])
    
    return file_name_list, anno


class CarotidSet(Dataset):
    def __init__(self, image_dir, json_path):
        super(CarotidSet, self).__init__()

        self.image_path, self.anno = json_parse(image_dir, json_path)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.resize = transforms.Resize(size=(128,128))  

    def __len__(self):
        return len(self.image_path)

    def make_gt(self, img_size, pt_list_x, pt_list_y):
        gt = Image.new(mode = "L", size = img_size)
        draw = ImageDraw.Draw(gt)
        prev_x = pt_list_x[0]
        prev_y = pt_list_y[0]
        for (x, y) in zip(pt_list_x, pt_list_y):
            draw.line((prev_x, prev_y) + (x,y), fill=255)
            prev_x = x
            prev_y = y
        return gt
    
    def __getitem__(self, idx):
        img = Image.open(self.image_path[idx])

        roi = self.anno[idx]['roi']
        li_x = self.anno[idx]['li']['x']
        li_y = self.anno[idx]['li']['y']
        ma_x = self.anno[idx]['ma']['x']
        ma_y = self.anno[idx]['ma']['y']

        img_li = self.make_gt(img.size, li_x, li_y)
        img_ma = self.make_gt(img.size, ma_x, ma_y)

        img_roi = img.crop((roi[0], roi[1], roi[0]+ roi[2], roi[1] + roi[3]))
        img_li = img_li.crop((roi[0], roi[1], roi[0]+ roi[2], roi[1] + roi[3]))
        img_ma = img_ma.crop((roi[0], roi[1], roi[0]+ roi[2], roi[1] + roi[3]))

        img_roi = self.resize(img_roi)
        img_li = self.resize(img_li)
        img_ma = self.resize(img_ma)

        threshold = 128
        img_li = img_li.point(lambda p: p > threshold and 255)
        img_ma = img_ma.point(lambda p: p > threshold and 255)

        img_roi = self.transform(img_roi)
        img_li = self.transform(img_li)
        img_ma = self.transform(img_ma)

        img_li = img_li.to(dtype=torch.bool).to(dtype=torch.float)
        img_ma = img_ma.to(dtype=torch.bool).to(dtype=torch.float)

        return (img_roi, img_li, img_ma)

if __name__ == '__main__':
    dataset = CarotidSet(image_dir='small_dataset', json_path='gTruth_pp_small.json')

    for i, (img, img_li, img_ma) in enumerate(dataset):
        print(img.shape)
        print(img_li.shape)
        print(img_ma.shape)
