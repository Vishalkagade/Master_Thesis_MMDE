import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2

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
                 ground_truth_paths,
                 transform,
                 mean = [0,0,0],
                 std = [1,1,1],
                 ):
        
        self.image_paths = image_paths
        self.mmde_map = mmde_path
        self.ground_truth_paths = ground_truth_paths
        self.radar_paths = radar_paths

        self.n_sample = len(self.image_paths)

        assert self.n_sample == len(self.ground_truth_paths)
        assert self.n_sample == len(self.mmde_map)
        # assert self.n_sample == len(self.radar_paths)
        self.final_transform = A.Compose([
        A.Normalize(mean=mean, std=std, max_pixel_value=255),
        ToTensorV2(),
        ])

        self.transform = transform

        self.radar_mde_transform = A.Compose([ToTensorV2()])

    def __len__(self):
        return self.n_sample

    def __getitem__(self, index):
        image = read_image(self.image_paths[index])
        height, width = image.shape[0], image.shape[1]
        if height == 720: # ZJU dataset
            image = image[720 // 3-10: 720 // 4 * 3+10,:,:]

        radar_img = load_sparse_depth(self.radar_paths[index])
        if height == 720: # ZJU dataset
            radar_img = radar_img[720 // 3-10: 720 // 4 * 3+10, :]

        ground_truth = load_sparse_depth(self.ground_truth_paths[index])
        if height == 720:
          ground_truth = ground_truth[720 // 3-10: 720 // 4 * 3+10, :]
        
        mmde_map = load_sparse_depth(self.mmde_map[index])
        max_depth = mmde_map.max()
        inverted_depth = max_depth - mmde_map
        if height == 720: # ZJU dataset
            mmde_map = inverted_depth[720 // 3-10: 720 // 4 * 3+10, :]

        

        if self.transform:
                transformed = self.transform(
                image=image,
                radar_img=radar_img,
                ground_truth=ground_truth,
                mmde_map=mmde_map
            )
                image = transformed['image']
                radar_img = transformed['radar_img']
                ground_truth = transformed['ground_truth']
                mmde_map = transformed['mmde_map']

        #image, radar_img, ground_truth, mmde_map = self.cutmix(image, radar_img, ground_truth, mmde_map)
        # Debug shapes
        
        #Normalize

        image = self.final_transform(image=image)['image']
        radar_img = radar_img/radar_img.max()
        radar_img = self.radar_mde_transform(image=radar_img)['image']
        ground_truth = self.radar_mde_transform(image=ground_truth)['image']
        mmde_map = mmde_map/mmde_map.max()
        mmde_map = self.radar_mde_transform(image=mmde_map)['image']

        #image, radar_img, ground_truth, mmde_map = self.cutmix(image, radar_img, ground_truth, mmde_map)

        # Convert all to float tensors
        image, mmde_map, radar_img, ground_truth = [
            T.float() for T in [image, mmde_map, radar_img, ground_truth]]

        return image, mmde_map, radar_img, ground_truth
    

class ValDataLoader(Dataset):
    def __init__(self, 
                 image_paths, 
                 mmde_path, 
                 radar_paths,
                 ground_truth_paths,
                 transform,
                 ):
        
        self.image_paths = image_paths
        self.mmde_map = mmde_path
        self.ground_truth_paths = ground_truth_paths
        self.radar_paths = radar_paths

        self.n_sample = len(self.image_paths)

        assert self.n_sample == len(self.ground_truth_paths)
        assert self.n_sample == len(self.mmde_map)
        # assert self.n_sample == len(self.radar_paths)

        #self.cutmix = CutMix(probability = 0.9)

        self.transform = transform

        self.radar_mde_transform = A.Compose([ToTensorV2()])

    def __len__(self):
        return self.n_sample

    def __getitem__(self, index):
        image = read_image(self.image_paths[index])
        height, width = image.shape[0], image.shape[1]
        if height == 720: # ZJU dataset
            image = image[720 // 3-10: 720 // 4 * 3+10,:,:]

        radar_img = load_sparse_depth(self.radar_paths[index])
        if height == 720: # ZJU dataset
            radar_img = radar_img[720 // 3-10: 720 // 4 * 3+10, :]

        ground_truth = load_sparse_depth(self.ground_truth_paths[index])
        if height == 720:
          ground_truth = ground_truth[720 // 3-10: 720 // 4 * 3+10, :]
        
        mmde_map = load_sparse_depth(self.mmde_map[index])
        max_depth = mmde_map.max()
        inverted_depth = max_depth - mmde_map
        if height == 720: # ZJU dataset
            mmde_map = inverted_depth[720 // 3-10: 720 // 4 * 3+10, :]

        if self.transform:
                transformed = self.transform(
                image=image,
                radar_img=radar_img,
                ground_truth=ground_truth,
                mmde_map=mmde_map
            )
                image = transformed['image']
                radar_img = transformed['radar_img']
                ground_truth = transformed['ground_truth']
                mmde_map = transformed['mmde_map']

        #image, radar_img, ground_truth, mmde_map = self.cutmix(image, radar_img, ground_truth, mmde_map)

        image = self.radar_mde_transform(image=image)['image']
        radar_img = self.radar_mde_transform(image=radar_img)['image']
        ground_truth = self.radar_mde_transform(image=ground_truth)['image']
        mmde_map = self.radar_mde_transform(image=mmde_map)['image']

        # # Convert all to float tensors
        # image, mmde_map, radar_img, ground_truth = [
        #     T.float() for T in [image, mmde_map, radar_img, ground_truth]]
        
        return image, mmde_map, radar_img, ground_truth
    

