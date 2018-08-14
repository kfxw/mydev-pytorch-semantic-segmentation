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
dataset_val = VOCValDataset(args)
loader_val = torchdata.DataLoader(dataset_val, batch_size=args.val_batch_size, shuffle=False, collate_fn=user_scattered_collate, num_workers=1, drop_last=False)
iterator_val = iter(loader_val)

nets = (net_encoder, net_decoder, crit)
segmentation_module.cuda(device = args.gpu_id[0])

# 5. validate
def evaluate(segmentation_module, loader, args):
    acc_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    cls_ious_meter = AverageMeter()
    cls_mean_iou_meter = AverageMeter()

    segmentation_module.eval()

    for i in range(len(loader)):
	batch_data = next(loader)[0]
        # process data
        seg_label = as_numpy(batch_data['seg_label'])

        with torch.no_grad():
            segSize = (seg_label.shape[1], seg_label.shape[2])
            pred = torch.zeros(1, args.num_class, segSize[0], segSize[1])

            # forward pass
            pred = segmentation_module(batch_data, segSize=segSize)
            _, preds = torch.max(pred.data.cpu(), dim=1)
            preds = as_numpy(preds.squeeze(0))

        # calculate accuracy
        acc, pix = accuracy(preds, seg_label, 255)
        intersection, union = intersectionAndUnion(preds, seg_label, args.num_class, 255)
        acc_meter.update(acc, pix)
        intersection_meter.update(intersection)
        union_meter.update(union)
	mean_iou = (intersection/union)[union!=0].mean()
        print('[{}] iter {}, accuracy: {}, mIoU: {}'
              .format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), i, acc, mean_iou))

    iou = intersection_meter.sum / (union_meter.sum + 1e-10)
    for i, _iou in enumerate(iou):
        print('[{}] class [{}], IoU: {}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), i, _iou))

    print('[Eval Summary]:')
    print('Mean IoU: {:.4}, Accuracy: {:.2f}%'
          .format(iou.mean(), acc_meter.average()*100))

    return iou, iou.mean()

(cls_ious, cls_mean_iou) = evaluate(segmentation_module, iterator_val, args)
print cls_ious, cls_mean_iou
