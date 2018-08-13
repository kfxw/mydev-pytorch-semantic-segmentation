import os, sys
sys.path.append('../')

import time, datetime
import random
import argparse
from distutils.version import LooseVersion
import torch
import torch.nn as nn
from data.voc.VOCDataset import VOCValDataset
from networks import VGGModelBuilder, ResNetModelBuilder, SegmentationModule
from utils import AverageMeter, accuracy, intersectionAndUnion
from lib.nn import UserScatteredDataParallel, user_scattered_collate, patch_replication_callback
from lib.utils import as_numpy, mark_volatile
import lib.utils.data as torchdata
import matplotlib.pyplot as plt
import numpy as np

solver_cfg_name = "test_solver_cfg.py"
solver_cfg_name = solver_cfg_name.strip().split('.')[-2]
exec('from '+solver_cfg_name+' import opt')
args = opt()
print("Input arguments:")
for key, val in vars(args).items():
	print("{:16} {}".format(key, val))

builder = VGGModelBuilder()
net_encoder = builder.build_encoder(
        arch=args.arch_encoder,
        weights='../'+args.weights_encoder)

p=net_encoder.state_dict()
fc6_pytorch = p['model.fc6.weight'].numpy()

import caffe
caffe.set_mode_gpu()
net = caffe.Net('trainval_vgg_voc.prototxt', 'vgg16_20M.caffemodel', caffe.TEST)
fc6_caffe = net.params['fc6'][0].data
print fc6_caffe[0,:2,:,:], fc6_caffe.shape
print fc6_pytorch[0,:2,:,:], fc6_pytorch.shape
