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
from data.voc.VOCDataset import VOCTrainDataset, VOCValDataset
from networks import VGGModelBuilder, ResNetModelBuilder, SegmentationModule
from utils import AverageMeter, accuracy, intersectionAndUnion, Logger
from lib.nn import UserScatteredDataParallel, user_scattered_collate, patch_replication_callback
from lib.utils import as_numpy, mark_volatile
import lib.utils.data as torchdata

import matplotlib.pyplot as plt


# train one epoch
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
        loss = loss/args.batch_size_per_gpu#.mean()
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
            print('[{}] '
		  'Epoch:[{}][{}/{}], Time(iter/data):{:.2f}/{:.2f}, '
                  'lr(en/de):{:.2e}/{:.2e}, '
                  'train acc: {:4.2f}, Loss: {:10.4f}'
                  .format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
			  epoch, i, args.epoch_iters,
                          batch_time.average(), data_time.average(),
                          args.running_lr_encoder, args.running_lr_decoder,
                          ave_acc.value(), ave_total_loss.value()))

            fractional_epoch = epoch - 1 + 1. * i / args.epoch_iters
            history['train']['epoch'].append(fractional_epoch)
            history['train']['loss'].append(loss.data[0])
            history['train']['train_acc'].append(acc.data[0])

        # adjust learning rate
        cur_iter = i + (epoch - 1) * args.epoch_iters
        adjust_learning_rate(optimizers, cur_iter, args)

	#p=segmentation_module.state_dict()['decoder.conv_last.weight'][1,1,:,:]
	#print p


def evaluate(segmentation_module, loader, args):
    acc_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()

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


def checkpoint(nets, history, args, epoch_num):
    print('[{}] Saving checkpoints...'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    (net_encoder, net_decoder, crit) = nets
    suffix_latest = 'epoch_{}.pth'.format(epoch_num)

    dict_encoder = net_encoder.state_dict()
    dict_decoder = net_decoder.state_dict()

    # dict_encoder_save = {k: v for k, v in dict_encoder.items() if not (k.endswith('_tmp_running_mean') or k.endswith('tmp_running_var'))}
    # dict_decoder_save = {k: v for k, v in dict_decoder.items() if not (k.endswith('_tmp_running_mean') or k.endswith('tmp_running_var'))}
    
    torch.save(history,
               '{}/history_{}'.format(args.snapshot_prefix, suffix_latest))
    torch.save(dict_encoder,
               '{}/encoder_{}'.format(args.snapshot_prefix, suffix_latest))
    torch.save(dict_decoder,
               '{}/decoder_{}'.format(args.snapshot_prefix, suffix_latest))


def group_weight(module, base_lr):
    group_decay = []
    group_no_decay = []
    group_no_decay_double_lr = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_no_decay_double_lr.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.conv._ConvNd):
            group_no_decay_double_lr.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay) + len(group_no_decay_double_lr)
    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0), dict(params=group_no_decay_double_lr, weight_decay=.0, lr=base_lr*2)]
    return groups


def create_optimizers(nets, args):
    (net_encoder, net_decoder, crit) = nets
    optimizer_encoder = torch.optim.SGD(
        group_weight(net_encoder, args.lr_encoder),
        lr=args.lr_encoder,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    optimizer_decoder = torch.optim.SGD(
        group_weight(net_decoder, args.lr_decoder),
        lr=args.lr_decoder,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    return (optimizer_encoder, optimizer_decoder)


def adjust_learning_rate(optimizers, cur_iter, args):
    scale_running_lr = ((1. - float(cur_iter) / args.max_iters) ** args.lr_pow)
    change_multi_encoder = args.lr_encoder * scale_running_lr / args.running_lr_encoder
    change_multi_decoder = args.lr_decoder * scale_running_lr / args.running_lr_decoder
    args.running_lr_encoder = args.lr_encoder * scale_running_lr
    args.running_lr_decoder = args.lr_decoder * scale_running_lr

    (optimizer_encoder, optimizer_decoder) = optimizers
    for param_group in optimizer_encoder.param_groups:
        param_group['lr'] *= change_multi_encoder
    for param_group in optimizer_decoder.param_groups:
        param_group['lr'] *= change_multi_decoder


def main(args):
    # Network Builders
    builder = VGGModelBuilder()
    net_encoder = builder.build_encoder(
        arch=args.arch_encoder,
        weights=args.weights_encoder)
    net_decoder = builder.build_decoder(
        arch=args.arch_decoder,
	fc_dim=1024,
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

    # Dataset and Loader
    dataset_train = VOCTrainDataset(args, batch_per_gpu=args.batch_size_per_gpu)
    loader_train = torchdata.DataLoader(
        dataset_train,
        batch_size=1,  # data_parallel have been modified, not useful
        shuffle=False,  # do not use this param
        collate_fn=user_scattered_collate,
        num_workers=1, # MUST be 1 or 0
        drop_last=True,
        pin_memory=True)

    print('[{}] 1 training epoch = {} iters'.format(args.epoch_iters, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    # create loader iterator
    iterator_train = iter(loader_train)

    # load nets into gpu
    if args.num_gpus > 1:
        segmentation_module = UserScatteredDataParallel(
            segmentation_module,
            device_ids=args.gpu_id)
        # For sync bn
        patch_replication_callback(segmentation_module)
    segmentation_module.cuda(device = args.gpu_id[0])

    # Set up optimizers
    nets = (net_encoder, net_decoder, crit)
    optimizers = create_optimizers(nets, args)

    # Main loop
    history = {'train': {'epoch': [], 'loss': [], 'train_acc': [], 'test_ious': [], 'test_mean_iou': []}}

    for epoch in range(args.start_epoch, args.num_epoch + 1):
	# test/validate
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
	if  epoch % args.test_epoch_interval == 0 or epoch == args.num_epoch:# epoch != 1 and
	    (cls_ious, cls_mean_iou) = evaluate(segmentation_module, iterator_val, args)
	    history['train']['test_ious'].append(cls_ious)
	    history['train']['test_mean_iou'].append(cls_mean_iou)
	else:
	    history['train']['test_ious'].append(-1)	# empty data
	    history['train']['test_mean_iou'].append(-1)

	# train
        train(segmentation_module, iterator_train, optimizers, history, epoch, args)

        # checkpointing
        checkpoint(nets, history, args, epoch)

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
    sys.stdout = Logger(args.log_file, sys.stdout)
    sys.stderr = Logger(args.log_file, sys.stderr)
    print('Model ID: {}'.format(args.id))

    print("[{}] Input arguments:".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))

    args.snapshot_prefix = os.path.join(args.snapshot_prefix, args.id)
    if not os.path.isdir(args.snapshot_prefix):
        os.makedirs(args.snapshot_prefix)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    main(args)
