import torch
import torch.nn as nn
import torchvision
from .ModelBuilder import ModelBuilder
from lib.nn import SynchronizedBatchNorm2d

class VGGModelBuilder(ModelBuilder):

    def build_encoder(self, arch='vgg16_20M', weights=''):
        pretrained = True if len(weights) == 0 else False
        if arch == 'vgg16_20M':
            net_encoder = VGG16_20M()
        #elif arch == 'vgg16_20M':
        #    net_encoder = VGG16_20M()
        else:
            raise Exception('Architecture undefined!')

        if len(weights) > 0:	# use pretrained model
            print('Loading weights for net_encoder')
            net_encoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
	else:			# init from sketch
	    net_encoder.apply(self.weights_init)
        return net_encoder


class VGG16_20M(nn.Module):
    def __init__(self):
        super(VGG16_20M, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,64,3, padding=1),
            nn.ReLU(inplace = True),
            nn.Conv2d(64,64,3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
	)
	self.conv2 = nn.Sequential(           
            nn.Conv2d(64,128,3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
	)
        self.conv3 = nn.Sequential(   
            nn.Conv2d(128,256,3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
	)
	self.conv4 = nn.Sequential(
            nn.Conv2d(256,512,3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=1, padding = 1),
	)
	# conv5
	self.conv5 = nn.Sequential(
            nn.Conv2d(512,512,3, padding=1, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3, padding=1, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3, padding=1, dilation=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=1,padding = 1),
        )
        self.fc = nn.Sequential(
            nn.Conv2d(512,1024,3,padding = 1, dilation=1),
            nn.ReLU (inplace=True),
            nn.Dropout(p=0.5),
            nn.Conv2d(1024,1024,1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
        )

    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.conv1(x); conv_out.append(x);
        x = self.conv2(x); conv_out.append(x);
        x = self.conv3(x); conv_out.append(x);
        x = self.conv4(x); conv_out.append(x);
        x = self.conv5(x); conv_out.append(x);
        x = self.fc(x); conv_out.append(x);

        if return_feature_maps:
            return conv_out
        return [x]
