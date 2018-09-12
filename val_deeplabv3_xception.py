# System libs
import os, sys
import time, datetime
# import math
import random
import argparse
from distutils.version import LooseVersion
# Numerical libs
import torch
import torch.nn as nn
# Our libs
from data.voc.VOCDataset_newstyle import VOCTrainDataset, VOCValDataset
from networks import XceptionModelBuilder, SegmentationModule
from utils import AverageMeter, accuracy, intersectionAndUnion, Logger
from lib.nn import UserScatteredDataParallel, user_scattered_collate, patch_replication_callback
from lib.utils import as_numpy, mark_volatile
import lib.utils.data as torchdata

import matplotlib.pyplot as plt
from scipy import misc
import numpy as np

def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.training = False

def evaluate(segmentation_module, loader, args):
    acc_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    cls_ious_meter = AverageMeter()
    cls_mean_iou_meter = AverageMeter()

    eval_model = segmentation_module
    eval_model.eval()

    f = file(args.val_list_file).readlines()

    for i in range(len(loader)):
	batch_data = next(loader)[0]

        # process data
        seg_label = as_numpy(batch_data['seg_label'])

        with torch.no_grad():
            segSize = (seg_label.shape[1], seg_label.shape[2])
            pred = torch.zeros(1, args.num_class, segSize[0], segSize[1])

            # forward pass
            pred = eval_model(batch_data, segSize=segSize)
            _, preds = torch.max(pred.data.cpu(), dim=1)
            preds = as_numpy(preds.squeeze(0))

	img = misc.imread(args.root_dataset+'/'+f[i].strip().split(' ')[0])
	misc.imsave('./tmp/'+f[i].strip().split(' ')[0].split('/')[-1].replace('.jpg','.png'), preds[:img.shape[0],:img.shape[1]].astype(np.uint8))

	preds = preds[:img.shape[0],:img.shape[1]]
	seg_label = seg_label[:,:img.shape[0],:img.shape[1]]

        # calculate accuracy
        acc, pix = accuracy(preds, seg_label, 255)
        intersection, union = intersectionAndUnion(preds, seg_label, args.num_class, 255)
        acc_meter.update(acc, pix)
        intersection_meter.update(intersection)
        union_meter.update(union)
        mean_iou = (intersection/(union+1e-10))[union!=0].mean()
        print('[{}] iter {}, accuracy: {:.5f}, mIoU: {:.5f}'
              .format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), i, acc, mean_iou))

    iou = intersection_meter.sum / (union_meter.sum + 1e-10)
    for i, _iou in enumerate(iou):
        print('[{}] class [{}], IoU: {}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), i, _iou))

    print('[{}] [Eval Summary]:'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    print('Mean IoU: {:.4}, Accuracy: {:.2f}%'
          .format(iou.mean(), acc_meter.average()*100))

    return iou, iou.mean()


def main(args):
    # Network Builders
    builder = XceptionModelBuilder()
    net_encoder = builder.build_encoder(
        arch=args.arch_encoder,
        weights=args.weights_encoder,
	overall_stride=args.overall_stride)
    net_decoder = builder.build_decoder(
        arch=args.arch_decoder,
	fc_dim=2048,
        num_class=args.num_class,
        weights=args.weights_decoder)

    crit = nn.CrossEntropyLoss(ignore_index=255, reduction='sum')

    if args.arch_decoder.endswith('deepsup'):
        segmentation_module = SegmentationModule(
            net_encoder, net_decoder, crit, args.deep_sup_scale)
    else:
        segmentation_module = SegmentationModule(
            net_encoder, net_decoder, crit)
    print(segmentation_module)

    segmentation_module.cuda()

    # Set up optimizers
    nets = (net_encoder, net_decoder, crit)
    dataset_val = VOCValDataset(args)	# create val dataset loader for every eval, in order to use drop_last=false
    loader_val = torchdata.DataLoader(
            dataset_val,
	    # data_parallel have been modified, MUST use val batch size
	    #   and collate_fn MUST be user_scattered_collate
            batch_size=args.val_batch_size,	
            shuffle=False,
            collate_fn=user_scattered_collate,
            num_workers=1, # MUST be 1 or 0
            drop_last=False)
    iterator_val = iter(loader_val)
    (cls_ious, cls_mean_iou) = evaluate(segmentation_module, iterator_val, args)

    print('[{}] Training Done!'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser()
    parser.add_argument('--solver_config', type=str, help='name of training/testing configuration file. cfg file has to be at CURRENT DIRECTORY and ends with .py extension')
    args = parser.parse_args()
    solver_cfg_name = args.solver_config.strip().split('.')[-2]
    exec('from '+solver_cfg_name+' import opt')

    args = opt()
    args.batch_size = args.num_gpus * args.batch_size_per_gpu
    args.max_iters = args.epoch_iters * args.num_epoch
    args.running_lr_encoder = args.lr_encoder
    args.running_lr_decoder = args.lr_decoder

    args.id += '_' + str(args.arch_encoder)
    args.id += '_' + str(args.arch_decoder)
    args.id += '_' + str(args.dataset_name)
    args.id += '_ngpus' + str(args.num_gpus)
    args.id += '_batchSize' + str(args.batch_size)
    args.id += '_trainCropSize' + str(args.train_cropSize)
    args.id += '_LR_encoder' + str(args.lr_encoder)
    args.id += '_LR_decoder' + str(args.lr_decoder)
    args.id += '_epoch' + str(args.num_epoch)
    args.id += '+' + datetime.datetime.now().strftime("%m%d-%H:%M:%S")	# add time stamp to avoid over writting
    args.log_file = args.id + '.log'
    print('Model ID: {}'.format(args.id))

    print("[{}] Input arguments:".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))

    args.weights_encoder = './trainingResults/deeplabv3_xception_allbn_8s_xception_deeplabv3_aspp_bilinear_VOC2012aug_ngpus5_batchSize15_trainCropSize504_LR_encoder2e-08_LR_decoder2e-08_epoch46+0905-19:07:38/encoder_epoch_46.pth'
    args.weights_decoder = './trainingResults/deeplabv3_xception_allbn_8s_xception_deeplabv3_aspp_bilinear_VOC2012aug_ngpus5_batchSize15_trainCropSize504_LR_encoder2e-08_LR_decoder2e-08_epoch46+0905-19:07:38/decoder_epoch_46.pth'
    args.val_batch_size = 1

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    main(args)
