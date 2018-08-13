import sys
sys.path.append('../')

import os
import time
import random
import argparse
from distutils.version import LooseVersion
import torch
import torch.nn as nn
from data.voc.VOCDataset import VOCValDataset
from networks import VGGModelBuilder, ResNetModelBuilder, SegmentationModule
from utils import AverageMeter
from lib.nn import UserScatteredDataParallel, user_scattered_collate, patch_replication_callback
import lib.utils.data as torchdata
import matplotlib.pyplot as plt
import numpy as np

class opt():
	def __init__(self):
		self.root_dataset = '/home/kfxw/Development/data/VOC/VOC2012-May-11-data/VOCdevkit/VOC2012'
		self.cropSize = 512	# 512,328
		self.random_flip = True
		self.random_scale = True
		self.random_scale_factor_max = 1.2
		self.random_scale_factor_min = 0.75
		self.train_list_file = '/home/kfxw/Development/data/VOC/VOC2012-May-11-data/VOCdevkit/VOC2012/ImageSets/Segmentation/deeplab_trainval_aug.txt'
		self.val_list_file = '/home/kfxw/Development/data/VOC/VOC2012-May-11-data/VOCdevkit/VOC2012/ImageSets/Segmentation/deeplab_val.txt'
		self.test_list_file = '/home/kfxw/Development/data/VOC/VOC2012-May-11-data/VOCdevkit/VOC2012/ImageSets/Segmentation/deeplab_val.txt'
		self.val_batch_size = 4

arg = opt()
dataset_val = VOCValDataset(arg)
loader_val = torchdata.DataLoader(dataset_val, batch_size=4, shuffle=False, collate_fn=user_scattered_collate, num_workers=1, drop_last=True, pin_memory=False)
iterator_val = iter(loader_val)
batch_data = next(iterator_val)
img = batch_data[0]['data'].cpu().numpy()
plt.imshow(img[1,::-1,:,:].transpose(1,2,0)/255.0+0.5)
plt.show()

batch_data = next(iterator_val)
img = batch_data[0]['data'].cpu().numpy()
plt.imshow(img[1,::-1,:,:].transpose(1,2,0)/255.0+0.5)
plt.show()

batch_data = next(iterator_val)
img = batch_data[0]['data'].cpu().numpy()
plt.imshow(img[1,::-1,:,:].transpose(1,2,0)/255.0+0.5)
plt.show()

batch_data = next(iterator_val)
img = batch_data[0]['data'].cpu().numpy()
plt.imshow(img[1,::-1,:,:].transpose(1,2,0)/255.0+0.5)
plt.show()

seg = batch_data[0]['seg_label'].cpu().numpy()
plt.imshow(seg[1,:,:])
plt.show()
