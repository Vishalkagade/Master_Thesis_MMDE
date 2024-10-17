import os
import glob
import torch
import torch.nn as nn
import cv2
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F

def read_image(path):
    img = cv2.imread(path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

def load_sparse_depth(input_sparse_depth_fp):
    input_sparse_depth = np.array(Image.open(input_sparse_depth_fp), dtype=np.float32)/256
    return input_sparse_depth


class TrainDataLoader(Dataset):
    def __init__(self,
                 image_paths,
                 mmde_path,
                 radar_paths,
                 ground_truth_paths,transform):

        self.image_paths = glob.glob(os.path.join(image_paths, '*.*'))
        self.mmde_map = glob.glob(os.path.join(mmde_path, '*.*'))
        self.ground_truth_paths = glob.glob(os.path.join(ground_truth_paths, '*.*'))
        self.radar_paths = glob.glob(os.path.join(radar_paths, '*.*'))

        self.n_sample = len(self.image_paths)

        assert self.n_sample == len(self.ground_truth_paths)
        assert self.n_sample == len(self.mmde_map)
        assert self.n_sample == len(self.radar_paths)

        self.transform = transform

    def __len__(self):
        return self.n_sample

    def __getitem__(self,index):

        image = read_image(self.image_paths[index])
        height, width = image.shape[0], image.shape[1]
        if height == 720: # ZJU dataset
            image = image[720 // 3-10: 720 // 4 * 3+10,:,:]
        image = image/255

        radar_img = load_sparse_depth(self.radar_paths[index])
        radar_img = radar_img/radar_img.max()
        if height == 720: # ZJU dataset
            radar_img = radar_img[720 // 3-10: 720 // 4 * 3+10, :]

        ground_truth = load_sparse_depth(self.ground_truth_paths[index])
        if height == 720:
          ground_truth = ground_truth[720 // 3-10: 720 // 4 * 3+10, :]
        #ground_truth = 1/(ground_truth+1e-8)

        mmde_map = load_sparse_depth(self.mmde_map[index])
        max_depth = mmde_map.max()
        inverted_depth = max_depth - mmde_map
        mmde_map = inverted_depth/inverted_depth.max()
        if height == 720: # ZJU dataset
            mmde_map = mmde_map[720 // 3-10: 720 // 4 * 3+10, :]

        if self.transform:
            image = self.transform(image)
            mmde_map = self.transform(mmde_map)
            ground_truth = self.transform(ground_truth)
            radar_img = self.transform(radar_img)
        image, mmde_map,radar_img, ground_truth = [T.float() for T in [image, mmde_map,radar_img, ground_truth]]


        return image, mmde_map,radar_img, ground_truth