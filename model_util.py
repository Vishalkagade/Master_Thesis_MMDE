import torch.nn as nn
import torch


# from convolutions import convbnrelu,conv1x1,conv3x3,batchnorm,InvertedResidualBlock,CRPBlock
def conv3x3(in_channels, out_channels, stride=1, dilation=1, groups=1, bias=False):
    """3x3 Convolution: Depthwise:
    https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=bias, groups=groups)


def conv1x1(in_channels, out_channels, stride=1, groups=1, bias=False,):
    "1x1 Convolution: Pointwise"
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=bias, groups=groups)


def batchnorm(num_features):
    """
    https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
    """
    return nn.BatchNorm2d(num_features, affine=True, eps=1e-5, momentum=0.1)
def convbnrelu(in_channels, out_channels, kernel_size = 3, stride=1, groups=1, act=True):
    "conv-batchnorm-relu"
    if act:
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=int(kernel_size / 2.), groups=groups, bias=False),
                             batchnorm(out_channels),
                             nn.ReLU6(inplace=True))
    else:
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size =  kernel_size, stride=stride, padding=int(kernel_size / 2.), groups=groups, bias=False),
                             batchnorm(out_channels))
    
class image_CBR_2(nn.Module):
    """
    Implementation with 2 CBR block with fusion
    """
    def __init__(self,in_channels,out_channels):
        super(image_CBR_2,self).__init__()

        self.cbs2 = nn.Sequential(convbnrelu(in_channels=in_channels,out_channels=out_channels),
                                    convbnrelu(in_channels=out_channels,out_channels=out_channels))
        self.max_pool = nn.MaxPool2d(kernel_size=2,stride=2,return_indices= True)

    def forward(self,x,depth_latent_feature):

        x = self.cbs2(x)
        x,indices = self.max_pool(x + depth_latent_feature)

        return x,indices
    
class image_CBR_3(nn.Module):
    """
    Implementation with 3 CBR block with fusion
    """
    def __init__(self,in_channels,out_channels,p = 0.1):
        super(image_CBR_3,self).__init__()

        self.cbs3 = nn.Sequential(convbnrelu(in_channels=in_channels,out_channels=out_channels),
                                    convbnrelu(in_channels=out_channels,out_channels=out_channels),#
                                    convbnrelu(in_channels=out_channels,out_channels=out_channels))
        self.max_pool = nn.MaxPool2d(kernel_size=2,stride=2,return_indices= True)
        self.dropout = nn.Dropout()

    def forward(self,x,depth_latent_feature):

        x = self.cbs3(x)
        x,indices = self.max_pool(x + depth_latent_feature)

        return self.dropout(x),indices
class depth_CBR_2(nn.Module):
    """
    Implementation with 2 CBR block with fusion
    """

    def __init__(self,in_channels,out_channels):
        super(depth_CBR_2,self).__init__()

        self.cbs2 = nn.Sequential(convbnrelu(in_channels=in_channels,out_channels=out_channels),
                                    convbnrelu(in_channels=out_channels,out_channels=out_channels))
        self.max_pool = nn.MaxPool2d(kernel_size=2,stride=2)

    def forward(self,x):
        skip = self.cbs2(x)

        x = self.max_pool(skip)
        return x,skip

class depth_CBR_3(nn.Module):
    """
    Implementation with 3 CBR block with fusion
    """
    def __init__(self,in_channels,out_channels,p = 0.1):
        super(depth_CBR_3,self).__init__()

        self.cbs3 = nn.Sequential(convbnrelu(in_channels=in_channels,out_channels=out_channels),
                                    convbnrelu(in_channels=out_channels,out_channels=out_channels),
                                    convbnrelu(in_channels=out_channels,out_channels=out_channels))
        self.max_pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.dropout = nn.Dropout()

    def forward(self,x):

        skip = self.cbs3(x)
        x = self.dropout(self.max_pool(skip))

        return x,skip

class dec_CBS3(nn.Module):#
    def __init__(self,in_channels,out_channels):
        super().__init__()

        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.cbs3 = nn.Sequential(convbnrelu(in_channels=in_channels,out_channels=out_channels),
                                    convbnrelu(in_channels=out_channels,out_channels=out_channels),#
                                    convbnrelu(in_channels=out_channels,out_channels=out_channels),
                                    nn.Dropout())
    def forward(self,x,indices,output_size):
        x = self.unpool(x,indices,output_size = output_size)
        return self.cbs3(x)

class dec_CBS2(nn.Module):#
    def __init__(self,in_channels,out_channels,dropout = True):
        super().__init__()

        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.cbs3 = nn.Sequential(convbnrelu(in_channels=in_channels,out_channels=out_channels),
                                    convbnrelu(in_channels=out_channels,out_channels=out_channels))
        if dropout:
            self.dropout = nn.Dropout()
    def forward(self,x,indices,output_size):
        x = self.unpool(x,indices,output_size)
        return self.cbs3(x)
    
class FullyConnected(torch.nn.Module):
    def __init__(self,
                    in_features,
                    out_features,
                    weight_initializer='kaiming_uniform',
                    activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                    dropout_rate=0.00):
            super(FullyConnected, self).__init__()

            self.fully_connected = torch.nn.Linear(in_features, out_features)

            if weight_initializer == 'kaiming_normal':
                torch.nn.init.kaiming_normal_(self.fully_connected.weight)
            elif weight_initializer == 'xavier_normal':
                torch.nn.init.xavier_normal_(self.fully_connected.weight)
            elif weight_initializer == 'xavier_uniform':
                torch.nn.init.xavier_uniform_(self.fully_connected.weight)

            self.activation_func = activation_func

            if dropout_rate > 0.00 and dropout_rate <= 1.00:
                self.dropout = torch.nn.Dropout(p=dropout_rate)
            else:
                self.dropout = None

    def forward(self, x):
        x = self.fully_connected(x)

        if self.activation_func is not None:
            x = self.activation_func(x)

        if self.dropout is not None:
            return self.dropout(x)
        else:
            return x