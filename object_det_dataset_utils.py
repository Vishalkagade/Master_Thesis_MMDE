import json
import numpy as np
import matplotlib.pyplot as plt
import os
import ast
from glob import glob
from tqdm.notebook import tqdm
import cv2

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
from torch.utils.data import Dataset, DataLoader

def get_mean_std(loader):
  """
  to get the normalise parameters for training  
  """
  channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0

  for image in tqdm(loader):
      if num_batches > 500:
          break
      channels_sum += torch.mean(image, dim=[0, 2, 3])
      channels_sqrd_sum += torch.mean(image ** 2, dim=[0, 2, 3])
      num_batches += 1

  mean = channels_sum / num_batches
  std = (channels_sqrd_sum / num_batches - mean ** 2) ** 0.5

  return mean, std
  

class MeanStdDataset(Dataset):
    def __init__(self, image_paths, transform):

        self.image_paths = image_paths
        self.transform = transform

    def read_image(self, path):
        return cv2.cvtColor(
            cv2.imread(path, cv2.IMREAD_COLOR),
            cv2.COLOR_BGR2RGB
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        # Load image
        image = self.read_image(self.image_paths[idx])

        # transform image
        image = self.transform(image=image)["image"]

        return image

bdd100k_to_kitti_map = {
    "car": ("Car", 0,),
    "pedestrian": ("Pedestrian", 1),
    "truck": ("Truck", 4),
    "rider": ("Rider", 7),
    "bus": ("Bus", 8),
    "train": ("Train", 9),
    "motorcycle": ("Motorcycle", 10),
    "bicycle": ("Bicycle", 11),
    "traffic sign" : ("Traffic-sign", 12),
    "traffic light" : ("Traffic-light", 13)
} # using kitty format to convert it faster

ignore_bdd100k_classes = ["other vehicle", "trailer" , "other person"]

def process(images_path, labels_path, output_dir):
  """
  to process the bdd100k dataset into kitty for object detection

  """

  with open(labels_path, mode="r") as f:
      labels = json.loads(f.read())

  image_file_names = set(os.listdir(images_path))
  label_file_names = {item["name"] for item in labels if "labels" in item}

  missing_labels = image_file_names.difference(label_file_names)
  label_file_names = list(image_file_names.difference(missing_labels))

  for file_name in missing_labels:
      file_path = os.path.join(images_path, file_name)
      if os.path.exists(file_path):
          os.remove(file_path)

  for item in tqdm(labels):

      file_name = item["name"]

      if file_name not in label_file_names:
          continue

      file = open(os.path.join(output_dir, file_name.replace(".jpg", ".txt")), mode="a")

      for _label in item["labels"]:
          if _label["attributes"]["occluded"] or \
              _label["attributes"]["truncated"] or \
              _label["category"] in ignore_bdd100k_classes:
              continue

          class_info = bdd100k_to_kitti_map[_label["category"]]
          class_name = class_info[0]
          class_id = class_info[1]

          xyxy = [
              str(_label["box2d"]["x1"]),
              str(_label["box2d"]["y1"]),
              str(_label["box2d"]["x2"]),
              str(_label["box2d"]["y2"])
          ]

          w_str = f"{class_name} {class_id} {' '.join(xyxy)}"
          file.write(w_str +"\n")

      file.close()