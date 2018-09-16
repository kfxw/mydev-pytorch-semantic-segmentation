import numpy as np
import torch
from torch import nn
from p_pooling import P_Pooling

class P_Pooling_Module(torch.nn.Module):
	def __init__(self, channel, kernel_size, stride=1, padding=1, init_bias=30, max_p=50, min_p=1, side_branch_scale=1):
		super(P_Pooling_Module, self).__init__()
		self.pooling_conv_p1 = nn.Conv2d(channel, channel, kernel_size, stride=stride, padding=padding, groups=channel, bias=False)
		self.pooling_relu_p1 = nn.PReLU(num_parameters=channel, init=0.25)
		self.pooling_conv_p2 = nn.Conv2d(channel, channel, 1, stride=1, padding=0, groups=channel, bias=True)
		self.pooling_relu_p2 = nn.Hardtanh(min_val=min_p-1e-10,max_val=max_p+1e-10, inplace=False)
		self.pooling = P_Pooling(kernel_size, stride=stride, padding=padding)
		self.init_bias = init_bias
		self.scale = side_branch_scale

		#self.max_pooling = nn.MaxPool2d(kernel_size, stride, 1)

	def forward(self, bottom, side_branch_scale=1):
		bottom_1 = bottom.clone() * self.scale
		bottom_1 = self.pooling_conv_p1(bottom_1)
		bottom_1 = self.pooling_relu_p1(bottom_1)
		bottom_1 = self.pooling_conv_p2(bottom_1)
		bottom_1 = self.pooling_relu_p2(bottom_1)
		bottom_1.detach()
		res = self.pooling(bottom, bottom_1)

		return res
