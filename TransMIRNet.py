import torch.nn as nn
from model import encoder_image,radar_encoder_sparse,mmde_encoder,association_decoder
from linear_attention import LocalFeatureTransformer


class fusion_net(nn.Module):
  def __init__(self,params):
    super(fusion_net, self).__init__()
    self.params = params

    self.encoder_image = encoder_image(params)
    self.encoder_radar = radar_encoder_sparse(params)
    self.mmde_encoder = mmde_encoder(params)

    self.attention = LocalFeatureTransformer(['self','cross'], n_layers=4, d_model=512)

    self.decoder = association_decoder(params,self.encoder_image.feat_out_channels, self.encoder_radar.feat_out_channels, self.mmde_encoder.feat_out_channels)

  def forward(self,image,radar_img,mmde_map):
    image_features = self.encoder_image(image)
    radar_features = self.encoder_radar(radar_img)
    mmde_features = self.mmde_encoder(mmde_map)
    radar_feature_reshape = radar_features[-1].view(radar_features[-1].shape[0],radar_features[-1].shape[1],-1).permute(0,2,1) #1*512*10*40
    image_feature_reshape = image_features[-1].view(image_features[-1].shape[0],image_features[-1].shape[1],-1).permute(0,2,1)
    latent_depth_tf, latent_image_pooled_tf = self.attention(radar_feature_reshape, image_feature_reshape)
    latent_depth_tf = latent_depth_tf.permute(0, 2, 1).view(radar_features[-1].shape)
    latent_image_pooled_tf = latent_image_pooled_tf.permute(0, 2, 1).view(image_features[-1].shape)
    radar_features[-1] = latent_depth_tf
    image_features[-1] = latent_image_pooled_tf
    depth_conf = self.decoder(image_features,radar_features,mmde_features)
    return depth_conf