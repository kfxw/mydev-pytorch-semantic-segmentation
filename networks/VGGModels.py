import torch
import torch.nn as nn
import torchvision
from .ModelBuilder import ModelBuilder
from lib.nn import SynchronizedBatchNorm2d
from collections import OrderedDict

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
            print('Loading weights for net_encoder: {}'.format(weights))
	    if weights.endswith('pkl'):
		pretrained = torch.load(weights, map_location=lambda storage, loc: storage).state_dict()
		pretrained = {'model.'+k: v for k, v in pretrained.items()}
		net_encoder.load_state_dict(pretrained, strict=False)
	    else:
        	net_encoder.load_state_dict(
                    torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
	else:			# init from sketch
	    net_encoder.apply(self.weights_init)
        return net_encoder


class VGG16_20M(nn.Module):
    def __init__(self):
        super(VGG16_20M, self).__init__()
        """self.conv1 = nn.Sequential(
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
        )"""
	self.model = nn.Sequential(OrderedDict([\
		('conv1_1',nn.Conv2d(3,64,3, padding=1)),\
		('relu1_1',nn.ReLU(inplace=True)),\
		('conv1_2',nn.Conv2d(64,64,3, padding=1)),\
		('relu1_2',nn.ReLU(inplace=True)),\
		('pool1',nn.MaxPool2d(2, stride=2, padding=1)),\
		('conv2_1',nn.Conv2d(64,128,3, padding=1)),\
		('relu2_1',nn.ReLU(inplace=True)),\
		('conv2_2',nn.Conv2d(128,128,3, padding=1)),\
		('relu2_2',nn.ReLU(inplace=True)),\
		('pool2',nn.MaxPool2d(2, stride=2, padding=1)),\
		('conv3_1',nn.Conv2d(128,256,3, padding=1)),\
		('relu3_1',nn.ReLU(inplace=True)),\
		('conv3_2',nn.Conv2d(256,256,3, padding=1)),\
		('relu3_2',nn.ReLU(inplace=True)),\
		('conv3_3',nn.Conv2d(256,256,3, padding=1)),\
		('relu3_3',nn.ReLU(inplace=True)),\
		('pool3',nn.MaxPool2d(2, stride=2, padding=1)),\
		('conv4_1',nn.Conv2d(256,512,3, padding=1)),\
		('relu4_1',nn.ReLU(inplace=True)),\
		('conv4_2',nn.Conv2d(512,512,3, padding=1)),\
		('relu4_2',nn.ReLU(inplace=True)),\
		('conv4_3',nn.Conv2d(512,512,3, padding=1)),\
		('relu4_3',nn.ReLU(inplace=True)),\
		('pool4',nn.MaxPool2d(2, stride=1, padding=0)),\
		('conv5_1',nn.Conv2d(512,512,3, padding=2, dilation=2)),\
		('relu5_1',nn.ReLU(inplace=True)),\
		('conv5_2',nn.Conv2d(512,512,3, padding=2, dilation=2)),\
		('relu5_2',nn.ReLU(inplace=True)),\
		('conv5_3',nn.Conv2d(512,512,3, padding=2, dilation=2)),\
		('relu5_3',nn.ReLU(inplace=True)),\
		('pool5',nn.MaxPool2d(3, stride=1, padding=1)),\
		('fc6',nn.Conv2d(512,1024,3, padding=1, dilation=1)),\
		('relu6',nn.ReLU(inplace=True)),\
		('drop6',nn.Dropout2d(p=0.5)),\
		('fc7',nn.Conv2d(1024,1024,1)),\
		('relu7',nn.ReLU(inplace=True)),\
		('drop7',nn.Dropout2d(p=0.5)),\
		('fc8_pascal',nn.Conv2d(1024,21,1)),\
	 ]))

    def forward(self, x, return_feature_maps=False):
        conv_out = []

	x = self.model.pool1(
		self.model.relu1_2(
		self.model.conv1_2(
		self.model.relu1_1(
		self.model.conv1_1(x)))))
	conv_out.append(x)
	x = self.model.pool2(
		self.model.relu2_2(
		self.model.conv2_2(
		self.model.relu2_1(
		self.model.conv2_1(x)))))
	conv_out.append(x)
	x = self.model.pool3(
		self.model.relu3_3(
		self.model.conv3_3(
		self.model.relu3_2(
		self.model.conv3_2(
		self.model.relu3_1(
		self.model.conv3_1(x)))))))
	conv_out.append(x)
	x = self.model.pool4(
		self.model.relu4_3(
		self.model.conv4_3(
		self.model.relu4_2(
		self.model.conv4_2(
		self.model.relu4_1(
		self.model.conv4_1(x)))))))
	conv_out.append(x)
	x = self.model.pool5(
		self.model.relu5_3(
		self.model.conv5_3(
		self.model.relu5_2(
		self.model.conv5_2(
		self.model.relu5_1(
		self.model.conv5_1(x)))))))
	conv_out.append(x)
	x = self.model.pool5(
		self.model.drop7(
		self.model.relu7(
		self.model.fc7(
		self.model.drop6(
		self.model.relu6(
		self.model.fc6(x)))))))
	conv_out.append(x)
        """x = self.conv1(x); conv_out.append(x);
        x = self.conv2(x); conv_out.append(x);
        x = self.conv3(x); conv_out.append(x);
        x = self.conv4(x); conv_out.append(x);
        x = self.conv5(x); conv_out.append(x);
        x = self.fc(x); conv_out.append(x);"""

        if return_feature_maps:
            return conv_out
        return [x]
