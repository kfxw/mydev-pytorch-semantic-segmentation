import sys
sys.path.append('../')

import os
import time
import random
import argparse
from distutils.version import LooseVersion
import torch
import torch.nn as nn
from data.voc.VOCDataset import VOCTrainDataset
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

arg = opt()
dataset_train = VOCTrainDataset(arg, batch_per_gpu = 4)
loader_train = torchdata.DataLoader(dataset_train, batch_size=4, shuffle=False, collate_fn=user_scattered_collate, num_workers=1, drop_last=True, pin_memory=True)
iterator_train = iter(loader_train)
batch_data = next(iterator_train)
img = batch_data[0]['data'].cpu().numpy()
plt.imshow(img[1,::-1,:,:].transpose(1,2,0)/255.0+0.5)
plt.show()

seg = batch_data[0]['seg_label'].cpu().numpy()
plt.imshow(seg[1,:,:])
plt.show()

plt.imshow(img[0,0,:,:])
