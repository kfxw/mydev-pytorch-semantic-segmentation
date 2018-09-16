import pdb
import torch
import torch.nn as nn
import torchvision
from lib.nn import SynchronizedBatchNorm2d

class SegmentationModuleBase(nn.Module):
    def __init__(self):
        super(SegmentationModuleBase, self).__init__()

    def pixel_acc(self, pred, label, ignore_label):
        _, preds = torch.max(pred, dim=1)
        valid = ((label >= 0) & (label != ignore_label)).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        return acc

class SegmentationModule(SegmentationModuleBase):
    def __init__(self, net_enc, net_dec, crit, deep_sup_scale=None):
        super(SegmentationModule, self).__init__()
        self.encoder = net_enc
        self.decoder = net_dec
        self.crit = crit
        self.deep_sup_scale = deep_sup_scale

    def forward(self, feed_dict, segSize=None):
	current_gpu = torch.cuda.current_device()
        if self.encoder.training is True: # training, only effected by .train() and .eval()
	    segSize = feed_dict['seg_label'][0].size()
            if self.deep_sup_scale is not None: # use deep supervision technique
                (pred, pred_deepsup) = self.decoder(self.encoder(feed_dict['data'].cuda(current_gpu),
					 return_feature_maps=True), segSize=segSize)
            else:
                pred = self.decoder(self.encoder(feed_dict['data'].cuda(current_gpu),
					 return_feature_maps=True), segSize=segSize)
            loss = self.crit(pred, feed_dict['seg_label'].cuda(current_gpu))
            if self.deep_sup_scale is not None:
                loss_deepsup = self.crit(pred_deepsup, feed_dict['seg_label'].cuda(current_gpu))
                loss = loss + loss_deepsup * self.deep_sup_scale

            acc = self.pixel_acc(pred, feed_dict['seg_label'].cuda(current_gpu), self.crit.ignore_index)
            return loss, acc
        else: # inference
            pred = self.decoder(self.encoder(feed_dict['data'].cuda(current_gpu), return_feature_maps=True), segSize=segSize)
	    #pred = torch.argmax(pred, dim=1)
            return pred

