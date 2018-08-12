import os, sys
sys.path.append('../')

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

# 1. cfg file
solver_cfg_name = "test_solver_cfg.py"
solver_cfg_name = solver_cfg_name.strip().split('.')[-2]
exec('from '+solver_cfg_name+' import opt')
args = opt()
print("Input arguments:")
for key, val in vars(args).items():
	print("{:16} {}".format(key, val))

# 2. network setup
builder = VGGModelBuilder()
net_encoder = builder.build_encoder(
        arch=args.arch_encoder,
        weights=args.weights_encoder)
net_decoder = builder.build_decoder(
        arch=args.arch_decoder,
        num_class=args.num_class,
	fc_dim = 1024,
        weights=args.weights_decoder)
crit = nn.CrossEntropyLoss(ignore_index=255)
segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)

# 3. dataset
dataset_train = VOCTrainDataset(args, batch_per_gpu=args.batch_size_per_gpu)
loader_train = torchdata.DataLoader(dataset_train, batch_size=1, shuffle=False, collate_fn=user_scattered_collate,
        num_workers=8, drop_last=True, pin_memory=True)
iterator_train = iter(loader_train)

# 4. preprocess for training
def group_weight(module):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.conv._ConvNd):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
    return groups

def create_optimizers(nets, args):
    (net_encoder, net_decoder, crit) = nets
    optimizer_encoder = torch.optim.SGD(
        group_weight(net_encoder),
        lr=args.lr_encoder,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    optimizer_decoder = torch.optim.SGD(
        group_weight(net_decoder),
        lr=args.lr_decoder,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    return (optimizer_encoder, optimizer_decoder)

nets = (net_encoder, net_decoder, crit)
optimizers = create_optimizers(nets, args)
segmentation_module.cuda(device = args.gpu_id[0])

args.running_lr_encoder = args.lr_encoder
args.running_lr_decoder = args.lr_decoder
args.batch_size = args.num_gpus * args.batch_size_per_gpu
args.max_iters = args.epoch_iters * args.num_epoch

# 5. train
def adjust_learning_rate(optimizers, cur_iter, args):
    scale_running_lr = ((1. - float(cur_iter) / args.max_iters) ** args.lr_pow)
    args.running_lr_encoder = args.lr_encoder * scale_running_lr
    args.running_lr_decoder = args.lr_decoder * scale_running_lr
    (optimizer_encoder, optimizer_decoder) = optimizers
    for param_group in optimizer_encoder.param_groups:
        param_group['lr'] = args.running_lr_encoder
    for param_group in optimizer_decoder.param_groups:
        param_group['lr'] = args.running_lr_decoder

def train(segmentation_module, iterator, optimizers, history, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    ave_total_loss = AverageMeter()
    ave_acc = AverageMeter()
    segmentation_module.train(True)	# affect BN and dropout
    # main loop
    tic = time.time()
    for i in range(args.epoch_iters):
        batch_data = next(iterator)[0]
        data_time.update(time.time() - tic)
        segmentation_module.zero_grad()
        # forward pass
        loss, acc = segmentation_module(batch_data)
        loss = loss.mean()
        acc = acc.mean()
        # Backward
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()
        # update average loss and acc
        ave_total_loss.update(loss.data[0])
        ave_acc.update(acc.data[0]*100)
        # calculate accuracy, and display
        if i % args.display_interval == 0:
            print('Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, '
                  'lr_encoder: {:.6f}, lr_decoder: {:.6f}, '
                  'Training accuracy: {:4.2f}, Loss: {:.6f}'
                  .format(epoch, i, args.epoch_iters,
                          batch_time.average(), data_time.average(),
                          args.running_lr_encoder, args.running_lr_decoder,
                          ave_acc.average(), ave_total_loss.average()))
            fractional_epoch = epoch - 1 + 1. * i / args.epoch_iters
            history['train']['epoch'].append(fractional_epoch)
            history['train']['loss'].append(loss.data[0])
            history['train']['train_acc'].append(acc.data[0])
        # adjust learning rate
        cur_iter = i + (epoch - 1) * args.epoch_iters
        adjust_learning_rate(optimizers, cur_iter, args)

history = {'train': {'epoch': [], 'loss': [], 'train_acc': [], 'test_ious': [], 'test_mean_iou': []}}
train(segmentation_module, iterator_train, optimizers, history, 0, args)
