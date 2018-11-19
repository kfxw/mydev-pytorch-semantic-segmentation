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
from data.voc.VOCDataset_newstyle import VOCTestDataset
from networks import XceptionModelBuilder, ResnetModelBuilder, SegmentationModule, XceptionPPoolingModelBuilder
from utils import AverageMeter, accuracy, intersectionAndUnion, Logger
from lib.nn import UserScatteredDataParallel, user_scattered_collate, patch_replication_callback
from lib.utils import as_numpy, mark_volatile
import lib.utils.data as torchdata

from scipy import misc
import numpy as np
import random

import matplotlib.pyplot as plt

def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.training = False

def test(segmentation_module, loader, args):
    eval_model = segmentation_module
    eval_model.eval()
    eval_model.apply(fix_bn)

    for i in range(len(loader)):
	batch_data = next(loader)[0]

        with torch.no_grad():
            segSize = (batch_data['data'].shape[2], batch_data['data'].shape[3])
            pred = torch.zeros(1, args.num_class, segSize[0], segSize[1])

            # forward pass
            pred = eval_model(batch_data, segSize=segSize)
	    if random.random() > 1:
		batch_data['data'] = torch.flip(batch_data['data'], [3])
        	pred += torch.flip(eval_model(batch_data, segSize=segSize), [3])
            _, preds = torch.max(pred.data.cpu(), dim=1)
            preds = as_numpy(preds.squeeze(0))[:-batch_data['pad_size'][0],:-batch_data['pad_size'][1]]

        print('iter {}/{}'.format(i, len(loader)))
	misc.imsave('./tmp/'+batch_data['img_name'].split('/')[-1].replace('.jpg','.png'), preds.astype(np.uint8))

    return

def main(args):
    # Network Builders
    builder = XceptionModelBuilder()	                       ## NOTE: need to be changed!!! ##
    net_encoder = builder.build_encoder(
        arch=args.arch_encoder,
        weights=args.weights_encoder,
	overall_stride=args.overall_stride)
    net_decoder = builder.build_decoder(
        arch=args.arch_decoder,
	fc_dim=2048,
        num_class=args.num_class,
        weights=args.weights_decoder)

    crit = None

    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
    print(segmentation_module)

    # Dataset and Loader
    dataset_test = VOCTestDataset(args)
    loader_test = torchdata.DataLoader(
        dataset_test,
        batch_size=1,  # data_parallel have been modified, not useful
        shuffle=False,  # do not use this param
        collate_fn=user_scattered_collate,
        num_workers=1, # MUST be 1 or 0
        drop_last=False,
        pin_memory=False)

    # create loader iterator
    iterator_test = iter(loader_test)

    segmentation_module.cuda()

    # Set up optimizers
    nets = (net_encoder, net_decoder, crit)

    test(segmentation_module, iterator_test, args)


    print('[{}] Test Done!'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser()
    parser.add_argument('--solver_config', type=str, help='name of training/testing configuration file. cfg file has to be at CURRENT DIRECTORY and ends with .py extension')
    args = parser.parse_args()
    solver_cfg_name = args.solver_config.strip().split('.')[-2]
    exec('from '+solver_cfg_name+' import opt')

    args = opt()
    args.weights_encoder = '/media/kfxw/Wei Data/deeplabv3_voc/train/deeplabv3_xception_allbn_1sttime_trainset_only_xception_deeplabv3_aspp_bilinear_VOC2012aug_ngpus1_batchSize16_trainCropSize504_LR_encoder2e-08_LR_decoder2e-08_epoch46+1110-08:07:39/encoder_epoch_43.pth'
    args.weights_decoder = '/media/kfxw/Wei Data/deeplabv3_voc/train/deeplabv3_xception_allbn_1sttime_trainset_only_xception_deeplabv3_aspp_bilinear_VOC2012aug_ngpus1_batchSize16_trainCropSize504_LR_encoder2e-08_LR_decoder2e-08_epoch46+1110-08:07:39/decoder_epoch_43.pth'

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    main(args)
