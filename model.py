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

from object_det_utils import autopad, Conv, DFL, make_anchors, dist2bbox

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
    self.upconv5 = upconv(feat_out_channels_img[4] + feat_out_channels_radar[4]+feat_out_channels_mmde[4], feat_out_channels_radar[4]//2) #
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
    self.upconv2    = upconv(feat_out_channels_img[1]+feat_out_channels_radar[1]+feat_out_channels_mmde[1]+feat_out_channels_radar[2]//2, feat_out_channels_radar[1]//2)
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

    final = torch.cat([img_final, rad_final, mmde_final], axis=1) # 1*1536*10*40
    
    upconv5 = self.upconv5(final) # 1536 --> 256
    upconv5 = self.bn5(upconv5)
    upconv5 = self.conv5(upconv5)
    upconv5 = torch.cat([img_skip3, rad_skip3, mmde_skip3, upconv5], axis=1) # 256+256+256+256 --> 1024

    upconv4_ = self.upconv4(upconv5) # 1024 --> 128
    upconv4 = self.bn4(upconv4_)
    upconv4 = self.conv4(upconv4)
    upconv4 = torch.cat([img_skip2, rad_skip2, mmde_skip2, upconv4], axis=1) # 128+128+128+128 --> 512

    upconv3_ = self.upconv3(upconv4) # 512 --> 64
    upconv3 = self.bn3(upconv3_)
    upconv3 = self.conv3(upconv3)
    upconv3 = torch.cat([img_skip1, rad_skip1, mmde_skip1, upconv3], axis=1) # 64+64+64+64 --> 256

    upconv2_ = self.upconv2(upconv3)
    upconv2 = self.bn2(upconv2_)
    upconv2 = self.conv2(upconv2)
    upconv2 = torch.cat([img_skip0, rad_skip0, mmde_skip0, upconv2], axis=1) # 64+64+64+64 --> 256

    upconv1 = self.upconv1(upconv2) # 256 --> 32
    upconv1 = self.bn1(upconv1)
    upconv1 = self.conv1(upconv1)

    final_depth = self.get_depth(upconv1) # 32 --> 1

    if self.params.return_object_detection == True:
      object_detecion_latent_feature = [upconv2, upconv3, upconv4] # 224, 256, 512
      return final_depth,object_detecion_latent_feature

    # depth = 90 * depth_conf[:, 0:1]
    # confidence = depth_conf[:, 1:2]

    return final_depth #confidence, depth,
   
class RCU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RCU, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.ReLU()

        self.maxpool = nn.MaxPool2d(kernel_size=4, stride=4)

    def forward(self, input):
        x = self.relu1(self.conv1(input))
        x = self.relu2(self.conv2(x))
       # x = x + self.maxpool(input)
        return x
    

class DetectionHead(nn.Module):
    """
    • This detection head module contains 3 heads, each takes input from
      different levels/ scales of light-weight refinenet network. Specifically [upconv2, upconv3, upconv4].
    • Each of the 3 heads are decoupled (i.e. seperate bounding box block and classification block)
    • This module is an implementation of YoloV8 head. Made changes to integrate it with the Multi-Task Network.

    Reference: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py

    """
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(
        self,
        num_classes=80,
        decoder_channels=(224, 256, 512),
        head_channels=(64, 128, 256),
        stride=(8, 16, 32),
        reg_max=16
    ):
        """
        Parameters:
        ----------

        num_classes: int
            The number of classes
        decoder_channels: tuple
            A tuple of decoder (LIGHT-WEIGHT REFINENET) out-channels at various layers l7, l5, l3 respectively
        head_channels: tuple
            A tuple of input-channels for the 3 heads of the detecion head
        stride: tuple
            A tuple of strides at different scales
        reg_max: int
            Anchor scale factor
        """

        super().__init__()

        self.num_classes = num_classes  # number of classes
        self.reg_max = reg_max # DFL channels (head_channels[0] // 16 to scale: 4)
        self.decoder_channels = decoder_channels
        self.num_heads = len(head_channels)  # num of detection heads
        self.no = self.num_classes + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.tensor(stride)  # strides will be computed during build using this (only once)

        self.reduce_spacial_dim = nn.ModuleList(RCU(in_channels = in_, out_channels= out_) for in_,out_ in zip(decoder_channels,head_channels))


        c2 = max((16, head_channels[0] // 4, self.reg_max * 4))
        c3 = max(head_channels[0], self.num_classes)

        self.bbox_layers = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1))
            for x in head_channels
        )   # [P3, P4, P5]

        self.class_layers = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.num_classes, 1))
            for x in head_channels
        )   # [P3, P4, P5]

        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):

        for i in range(self.num_heads):
            xin = self.reduce_spacial_dim[i](x[i])  # reduce spacial dimensions of the input
            box_out = self.bbox_layers[i](xin)  # box_out_channels = 4 * self.reg_max = 16
            cls_out = self.class_layers[i](xin) # cls_out_channels = no of classes
            x[i] = torch.cat((box_out, cls_out), 1) # N_CHANNELS = self.no = box_out_channels + cls_out_channels

        if self.training:
            # For input imahe height = 192, width = 640
            # x[0] shape = (batch_size, self.no, 40, 80)
            # x[1] shape = (batch_size, self.no, 12, 40)
            # x[2] shape = (batch_size, self.no, 6, 20)
            return x

        # N_OUT = (6 * 20) + (12 * 40) + (24 * 80) = 2520

        shape = x[0].shape  # BCHW
        if self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape
            # shape: self.anchors = (2, N_OUT); self.strides = (1, N_OUT)

        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2) # shape: (batch_size, self.no, N_OUT)

        box, cls = x_cat.split((self.reg_max * 4, self.num_classes), 1)
        # shape: box = (batch_size, 4 * self.reg_max, N_OUT); cls = (batch_size, no. of classes, N_OUT)

        dbox = dist2bbox(
            self.dfl(box),
            self.anchors.unsqueeze(0),
            xywh=True,
            dim=1
        ) * self.strides

        # shape: dbox = (batch_size, 4, N_OUT)

        y = torch.cat((dbox, cls.sigmoid()), 1)  # shape: (batch_size, 4 + no. of classes, N_OUT). e.g (2, 11, 2520)

        return y, x