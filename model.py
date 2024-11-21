import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import torch.nn as nn
import cv2

from convolutions import SparseConv,upconv

class mmde_encoder(nn.Module):
    def __init__(self,params):
        super(mmde_encoder,self).__init__()

        self.params = params
        if params.mmde_encoder == 'resnet34':
            self.base_model = models.resnet34(pretrained=True)
            self.base_model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) 
            self.feat_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
            self.feat_out_channels = [64, 64, 128, 256, 512]
        elif params.mmde_encoder == 'resnet18':
            self.base_model = models.resnet18(pretrained=True)
            self.base_model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) 
            self.feat_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
            self.feat_out_channels = [64, 64, 128, 256, 512]
        else:
            print('Not supported encoder: {}'.format(params.encoder))
    def forward(self, x):
        feature = x
        skip_feat = []
        i = 1
        for k, v in self.base_model._modules.items():
            if 'fc' in k or 'avgpool' in k:
                continue
            feature = v(feature)
            if self.params.encoder == 'mobilenetv2_bts':
                if i == 2 or i == 4 or i == 7 or i == 11 or i == 19:
                    skip_feat.append(feature)
            else:
                if any(x in k for x in self.feat_names):
                    skip_feat.append(feature)
            i = i + 1

        return skip_feat

class radar_encoder_sparse(nn.Module):
    def __init__(self,params):
        super(radar_encoder_sparse, self).__init__()

        self.params = params
        self.sparse_conv1 = SparseConv(params.radar_input_channels, 16, 7, activation='elu')
        self.sparse_conv2 = SparseConv(16, 16, 5, activation='elu')
        self.sparse_conv3 = SparseConv(16, 16, 3, activation='elu')
        self.sparse_conv4 = SparseConv(16, 3, 3, activation='elu')

        if params.encoder_radar == 'resnet34':
            self.base_model_radar = models.resnet34(pretrained=True)
            self.feat_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
            self.feat_out_channels = [64, 64, 128, 256, 512]
        elif params.encoder_radar == 'resnet18':
            self.base_model_radar = models.resnet18(pretrained=True)
            self.feat_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
            self.feat_out_channels = [64, 64, 128, 256, 512]
        else:
            print('Not supported encoder: {}'.format(params.encoder))

    def forward(self, x):
        mask = (x[:, 0] > 0).float().unsqueeze(1)
        feature = x
        feature, mask = self.sparse_conv1(feature, mask)
        feature, mask = self.sparse_conv2(feature, mask)
        feature, mask = self.sparse_conv3(feature, mask)
        feature, mask = self.sparse_conv4(feature, mask)

        skip_feat = []
        i = 1
        for k, v in self.base_model_radar._modules.items():
            if 'fc' in k or 'avgpool' in k:
                continue
            feature = v(feature)
            if any(x in k for x in self.feat_names):
                skip_feat.append(feature)
            i = i + 1
        return skip_feat
    
class encoder_image(nn.Module):
  def __init__(self,params):
    super(encoder_image, self).__init__()
    self.params = params

    if params.encoder == 'resnet34':
      self.base_model = models.resnet34(pretrained=True)
      self.feat_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
      self.feat_out_channels = [64, 64, 128, 256, 512]
    else:
        print('Not supported encoder: {}'.format(params.encoder))
  def forward(self, x):
    feature = x
    skip_feat = []
    i = 1
    for k, v in self.base_model._modules.items():
        if 'fc' in k or 'avgpool' in k:
            continue
        feature = v(feature)
        if self.params.encoder == 'mobilenetv2_bts':
            if i == 2 or i == 4 or i == 7 or i == 11 or i == 19:
                skip_feat.append(feature)
        else:
            if any(x in k for x in self.feat_names):
                skip_feat.append(feature)
        i = i + 1
    return skip_feat
  

class association_decoder(nn.Module):

  def __init__(self, params, feat_out_channels_img, feat_out_channels_radar, feat_out_channels_mmde):
    super(association_decoder, self).__init__()

    self.params = params
    #first upsamsling
    self.upconv5 = upconv(feat_out_channels_img[4] + feat_out_channels_radar[4]+feat_out_channels_mmde[4], feat_out_channels_radar[4]//2) #[64, 64, 128, 256, 512]
    self.bn5 = nn.BatchNorm2d(feat_out_channels_radar[4]//2, momentum=0.01, affine=True, eps=1.1e-5)
    self.conv5 = torch.nn.Sequential(nn.Conv2d(feat_out_channels_radar[4]//2, feat_out_channels_radar[4]//2, 3, 1, 1, bias=False),
                                          nn.ELU())
    #second upsampling
    self.upconv4    = upconv(feat_out_channels_img[3]+feat_out_channels_radar[3]+feat_out_channels_mmde[3]+feat_out_channels_radar[4]//2, feat_out_channels_radar[3]//2)
    self.bn4        = nn.BatchNorm2d(feat_out_channels_radar[3]//2, momentum=0.01, affine=True, eps=1.1e-5)
    self.conv4      = torch.nn.Sequential(nn.Conv2d(feat_out_channels_radar[3]//2, feat_out_channels_radar[3]//2, 3, 1, 1, bias=False),
                                          nn.ELU())

    #third upsampling
    self.upconv3    = upconv(feat_out_channels_img[2]+feat_out_channels_radar[2]+feat_out_channels_mmde[2]+feat_out_channels_radar[3]//2, feat_out_channels_radar[2]//2)
    self.bn3        = nn.BatchNorm2d(feat_out_channels_radar[2]//2, momentum=0.01, affine=True, eps=1.1e-5)
    self.conv3      = torch.nn.Sequential(nn.Conv2d(feat_out_channels_radar[2]//2, feat_out_channels_radar[2]//2, 3, 1, 1, bias=False),
                                          nn.ELU())
    #forth upsmapling
    self.upconv2    = upconv(feat_out_channels_img[1]+feat_out_channels_radar[1]+feat_out_channels_mmde[1]+feat_out_channels_radar[2]//2, feat_out_channels_radar[1])
    self.bn2        = nn.BatchNorm2d(feat_out_channels_radar[1]//2, momentum=0.01, affine=True, eps=1.1e-5)
    self.conv2      = torch.nn.Sequential(nn.Conv2d(feat_out_channels_radar[1]//2, feat_out_channels_radar[1]//2, 3, 1, 1, bias=False),
                                      nn.ELU())
    #fifth upsampling
    self.upconv1    = upconv(feat_out_channels_img[0]+feat_out_channels_radar[0]+feat_out_channels_mmde[0]+feat_out_channels_radar[1]//2, feat_out_channels_radar[0]//2)
    self.bn1        = nn.BatchNorm2d(feat_out_channels_radar[0]//2, momentum=0.01, affine=True, eps=1.1e-5)
    self.conv1      = torch.nn.Sequential(nn.Conv2d(feat_out_channels_radar[0]//2, feat_out_channels_radar[0]//2, 3, 1, 1, bias=False),
                                          nn.ELU())
    #last layer
    self.get_depth  = torch.nn.Sequential(nn.Conv2d(feat_out_channels_radar[0]//2, 1, 3, 1, 1, bias=False),
                                      nn.ReLU())

  def forward(self, image_features, radar_features, mmde_features):

    img_skip0, img_skip1, img_skip2, img_skip3, img_final = image_features[0], image_features[1], image_features[2], image_features[3], image_features[4] #[64, 64, 128, 256, 512]
    rad_skip0, rad_skip1, rad_skip2, rad_skip3, rad_final = radar_features[0], radar_features[1], radar_features[2], radar_features[3], radar_features[4]
    mmde_skip0, mmde_skip1, mmde_skip2, mmde_skip3, mmde_final = mmde_features[0], mmde_features[1], mmde_features[2], mmde_features[3], mmde_features[4]

    final = torch.cat([img_final, rad_final, mmde_final], axis=1) # 1*1536*10*24
    
    upconv5 = self.upconv5(final) # 1536 --> 256
    upconv5 = self.bn5(upconv5)
    upconv5 = self.conv5(upconv5)
    upconv5 = torch.cat([img_skip3, rad_skip3, mmde_skip3, upconv5], axis=1) # 256+256+256+256 --> 1024

    upconv4 = self.upconv4(upconv5) # 1024 --> 128
    upconv4 = self.bn4(upconv4)
    upconv4 = self.conv4(upconv4)
    upconv4 = torch.cat([img_skip2, rad_skip2, mmde_skip2, upconv4], axis=1) # 128+128+128+128 --> 512

    upconv3 = self.upconv3(upconv4) # 512 --> 64
    upconv3 = self.bn3(upconv3)
    upconv3 = self.conv3(upconv3)
    upconv3 = torch.cat([img_skip1, rad_skip1, mmde_skip1, upconv3], axis=1) # 64+64+64+64 --> 256

    upconv2 = self.upconv2(upconv3)
    upconv2 = self.bn2(upconv2)
    upconv2 = self.conv2(upconv2)
    upconv2 = torch.cat([img_skip0, rad_skip0, mmde_skip0, upconv2], axis=1) # 64+64+64+64 --> 256

    upconv1 = self.upconv1(upconv2) # 256 --> 32
    upconv1 = self.bn1(upconv1)
    upconv1 = self.conv1(upconv1)

    # confidence = self.get_depth(upconv1)
    # depth = self.params.max_depth * confidence
    depth_conf = self.get_depth(upconv1) # 32 --> 1
    # depth = 90 * depth_conf[:, 0:1]
    # confidence = depth_conf[:, 1:2]

    return depth_conf #confidence, depth,