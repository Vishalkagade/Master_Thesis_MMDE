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

import numpy as np
from glob import glob

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
from torch.utils.data import Dataset, DataLoader
import os

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
        #assert self.n_sample == len(self.radar_paths)

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


class BDD100KDataset(Dataset):
    def __init__(
        self,
        inputs,
        targets,
        transform,
        class_map,
        image_height,
        image_width,
        mean=[0., 0., 0.],
        std=[1., 1., 1.]
    ):
        
        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.class_map = class_map
        self.image_height = image_height
        self.image_width = image_width

        self.final_transform = A.Compose([
                A.Normalize(mean=mean, std=std, max_pixel_value=255),
                ToTensorV2(),
        ])
    
    def read_image(self, path):
        return cv2.cvtColor(
            cv2.imread(path, cv2.IMREAD_COLOR), 
            cv2.COLOR_BGR2RGB
        )
    
    def read_label(self, path):
        with open(path, 'r') as f:
            labels = f.read().splitlines()
        return labels
    
    def get_class_ids_and_bboxes(self, labels):
        # Convert the list to a NumPy array
        arr = np.array([line.split() for line in labels])

        # Extract class indices using class_map
        class_ids = np.array([self.class_map[class_name] for class_name in arr[:, 0]], dtype=int)

        # Extract xyxy coordinates
        bboxes = arr[:, 2:6].astype(float)
                                 
        return class_ids.tolist(), bboxes.tolist()
    
    def pascal_voc_to_yolo(self, bboxes):
        
        for i in range(len(bboxes)):
            xmin, ymin, xmax, ymax = bboxes[i]

            # Calculate center coordinates
            x_center = (xmin + xmax) / 2
            y_center = (ymin + ymax) / 2

            # Calculate width and height
            width = xmax - xmin
            height = ymax - ymin

            # Normalize coordinates and dimensions
            x_center /= self.image_width
            y_center /= self.image_height
            width /= self.image_width
            height /= self.image_height

            bboxes[i] = [x_center, y_center, width, height]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        
        # Load image and label
        image = self.read_image(self.inputs[idx])
        labels = self.read_label(self.targets[idx])

        if not labels:
            # if file is empty, get random image and labels
            random_idx = np.random.randint(0, len(self.inputs))
            return self.__getitem__(random_idx)
        
        # get class_ids and bboxes
        class_ids, bboxes = self.get_class_ids_and_bboxes(labels)

        if not bboxes:
            # if image has 0 bboxes, get random image and labels
            random_idx = np.random.randint(0, len(self.inputs))
            return self.__getitem__(random_idx)

        # preprocess
        aug = self.transform(image=image, bboxes=bboxes, category_ids=class_ids)
        image = aug["image"]
        bboxes = aug["bboxes"]
        class_ids = aug["category_ids"]
        
        if not bboxes:
            # after preprocess; if image has 0 bboxes, get random image and labels
            random_idx = np.random.randint(0, len(self.inputs))
            return self.__getitem__(random_idx)
        
        # final transform
        image = self.final_transform(image=image)["image"]

        # convert pascal_voc bboxes (xyxy) to yolo bboxes (xn,yn,wn,hn : normalized)
        self.pascal_voc_to_yolo(bboxes)
        
        # Combine class indices and bboxes coordinates
        target = torch.column_stack([
            torch.tensor(class_ids, dtype=torch.float32),
            torch.tensor(bboxes, dtype=torch.float32)
        ]) # shape: (num_bboxes, 5); [class_index, xn, yn, wn, hn]

        return image, target