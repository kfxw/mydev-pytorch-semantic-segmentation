import torch
import torch.nn as nn
import torchvision
from .ModelBuilder import ModelBuilder
from lib.nn import SynchronizedBatchNorm2d
from collections import OrderedDict

class XceptionModelBuilder(ModelBuilder):

    def build_encoder(self, arch='xception', weights='', overall_stride=32):
        pretrained = True if len(weights) == 0 else False
        if arch == 'xception':
            net_encoder = Xception(overall_stride=overall_stride)
        else:
            raise Exception('Architecture undefined!')

	net_encoder.apply(self.weights_init)
        if len(weights) > 0:	# use pretrained model
            net_encoder.load_state_dict(
                    torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_encoder

    # specialized weights initialization
    def weights_init(self, m):
        classname = m.__class__.__name__
	if classname.find('SeparableConv2d') != -1:
	    nn.init.kaiming_normal_(m.conv1.weight.data)
	    nn.init.kaiming_normal_(m.pointwise.weight.data)
        elif classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)


class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self,in_filters,out_filters,reps,strides=1,start_with_relu=True,grow_first=True,dilation=1):
        super(Block, self).__init__()

        if out_filters != in_filters or strides!=1:
            self.skip = nn.Conv2d(in_filters,out_filters,1,stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip=None
        
        self.relu = nn.ReLU(inplace=True)
        rep=[]

        filters=in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=dilation,bias=False,dilation=dilation))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters,filters,3,stride=1,padding=dilation,bias=False,dilation=dilation))
            rep.append(nn.BatchNorm2d(filters))
        
        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=dilation,bias=False,dilation=dilation))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3,strides,1))
        self.rep = nn.Sequential(*rep)

    def forward(self,inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x+=skip
        return x


class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """
    def __init__(self, overall_stride=32):
        super(Xception, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(3, 32, 3,2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32,64,3,bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        #do relu here

	dilation = 1
        self.block1=Block(64,128,2,2,start_with_relu=False,grow_first=True)	#block(in_dim,out_dim,reps,stride)
        self.block2=Block(128,256,2,2,start_with_relu=True,grow_first=True)
	if overall_stride == 8:
	    dilation *= 2
            self.block3=Block(256,728,2,1,start_with_relu=True,grow_first=True)	# pooling is at end of the block
	else:
	    self.block3=Block(256,728,2,2,start_with_relu=True,grow_first=True)

        self.block4=Block(728,728,3,1,start_with_relu=True,grow_first=True,dilation=dilation)
        self.block5=Block(728,728,3,1,start_with_relu=True,grow_first=True,dilation=dilation)
        self.block6=Block(728,728,3,1,start_with_relu=True,grow_first=True,dilation=dilation)
        self.block7=Block(728,728,3,1,start_with_relu=True,grow_first=True,dilation=dilation)

        self.block8=Block(728,728,3,1,start_with_relu=True,grow_first=True,dilation=dilation)
        self.block9=Block(728,728,3,1,start_with_relu=True,grow_first=True,dilation=dilation)
        self.block10=Block(728,728,3,1,start_with_relu=True,grow_first=True,dilation=dilation)
        self.block11=Block(728,728,3,1,start_with_relu=True,grow_first=True,dilation=dilation)

	if overall_stride == 8 or overall_stride == 16:
	    dilation *= 2
            self.block12=Block(728,1024,2,1,start_with_relu=True,grow_first=False,dilation=dilation)
	else:
	    self.block12=Block(728,1024,2,2,start_with_relu=True,grow_first=False,dilation=dilation)

        self.conv3 = SeparableConv2d(1024,1536,3,1,dilation,dilation=dilation)	# SeparableConv2d(in_dim,out_dim,kernel,stride,pad)
        self.bn3 = nn.BatchNorm2d(1536)

        #do relu here
        self.conv4 = SeparableConv2d(1536,2048,3,1,dilation,dilation=dilation)
        self.bn4 = nn.BatchNorm2d(2048)

    def forward(self, input, return_feature_maps=False):
        conv_out = []

        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
	conv_out.append(x)
        
        x = self.block1(x)
	conv_out.append(x)

        x = self.block2(x)
	conv_out.append(x)

        x = self.block3(x)
	conv_out.append(x)

        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
	conv_out.append(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
	conv_out.append(x)

        if return_feature_maps:
            return conv_out
        return [x]
