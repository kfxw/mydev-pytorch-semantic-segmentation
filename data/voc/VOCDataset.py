import os
import json
import torch
import torch.nn.functional as F
import lib.utils.data as torchdata
import cv2
from torchvision import transforms
import torchvision.transforms.functional as F1
from scipy.misc import imread, imresize
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy import misc

# Round x to the nearest multiple of p and x' >= x
def round2nearest_multiple(x, p):
    return ((x - 1) // p + 1) * p

class VOCTrainDataset(torchdata.Dataset):
    def __init__(self, opt, max_sample=-1, batch_per_gpu=1):
        self.root_dataset = opt.root_dataset
        self.cropSize = opt.cropSize
        self.random_flip = opt.random_flip
	self.random_scale = opt.random_scale
	self.random_scale_factor_min = 1
	self.random_scale_factor_max = 1
	if self.random_scale == True:
	    self.random_scale_factor_min = opt.random_scale_factor_min
	    self.random_scale_factor_max = opt.random_scale_factor_max
	    assert self.random_scale_factor_max > self.random_scale_factor_min
        # max down sampling rate of network to avoid rounding during conv or pooling
        # down sampling rate of segm labe
        #self.segm_downsampling_rate = opt.segm_downsampling_rate
        self.batch_per_gpu = batch_per_gpu

        # collect image per batch
        self.batch_record_list = []

        # override dataset length when trainig with batch_per_gpu > 1
        self.cur_idx = 0

        # mean and std, in BGR order
        self.img_transform = transforms.Compose([ transforms.Normalize(mean=[104.008, 116.669, 122.675], std=[1., 1., 1.]) ])
	self.mean = [104.008, 116.669, 122.675]
	self.std = [1., 1., 1.]

	# load img and seg list
        self.list_sample = open(opt.train_list_file, 'r').readlines()

        self.if_shuffled = False		# fixed setting, always to shuffle
	# select first 'max_sample' samples and get the number of samples
        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]
        self.num_sample = len(self.list_sample)
        assert self.num_sample > 0
        print '# samples: {}'.format(self.num_sample)

    def _get_sub_batch(self):
        while True:
            # get a sample record
            this_sample = self.list_sample[self.cur_idx]
            self.batch_record_list.append(this_sample.strip())

            # update current sample pointer
            self.cur_idx += 1
            if self.cur_idx >= self.num_sample:
                self.cur_idx = 0
                np.random.shuffle(self.list_sample)

            if len(self.batch_record_list) == self.batch_per_gpu:
                batch_records = self.batch_record_list
                self.batch_record_list = []
                break
        return batch_records

    def __getitem__(self, index):
        # NOTE: random shuffle for the first time. shuffle in __init__ is useless
        if not self.if_shuffled:
            np.random.shuffle(self.list_sample)
            self.if_shuffled = True

        # get sub-batch candidates
        batch_records = self._get_sub_batch()

        batch_images = torch.zeros(self.batch_per_gpu, 3, self.cropSize, self.cropSize)
        batch_segms = torch.zeros(self.batch_per_gpu, self.cropSize, self.cropSize).long()

        for i in range(self.batch_per_gpu):
            this_record = batch_records[i]

            # 1 load image and label
            image_path = os.path.normpath(self.root_dataset +  '/' + this_record.split(' ')[0])
            segm_path = os.path.normpath(self.root_dataset +  '/' + this_record.split(' ')[1])
            img = imread(image_path, mode='RGB')
            segm = imread(segm_path)

            assert(img.ndim == 3)
            assert(segm.ndim == 2)
            assert(img.shape[0] == segm.shape[0])
            assert(img.shape[1] == segm.shape[1])

	    # 2 random flip
            if self.random_flip == True:
                random_flip = np.random.choice([0, 1])
                if random_flip == 1:
                    img = cv2.flip(img, 1)
                    segm = cv2.flip(segm, 1)

            # 3 random scale
	    if self.random_scale == True:
		scale_factor = np.random.uniform(self.random_scale_factor_min, self.random_scale_factor_max)
		img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
		segm = cv2.resize(segm, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)

	    [img_height, img_width, _] = img.shape

	    # 4 RGB to BGR
            img = img.astype(np.float32)[:, :, ::-1] 

	    # 5 transpose from HxWxC to CxHxW
            img = img.transpose(2,0,1)

	    # 6 substract mean and normalize std
            img = self.img_transform(torch.from_numpy(img.copy()))#, self.mean, self.std)

            # 7 random crop or pad to 'cropSize'
            cropBeginIdx_h = 0
            cropBeginIdx_w = 0
	    if self.cropSize > img_height:	# vertical
		# if image size is smaller than crop size, then pad image to crop size
		padSize = self.cropSize - img_height
		img = F.pad(img, (0,0,0,padSize,0,0), 'constant', 0)
		segm = np.pad(segm, ((0, padSize), (0,0)), 'constant', constant_values=255)
		cropBeginIdx_h = 0
	    else:
		# if image size is larger than crop size, then randomly crop subimage out
		cropBeginIdx_h = random.randint(0, img_height-self.cropSize)
	    if self.cropSize > img_width:	# horizon
		# pad
		padSize = self.cropSize - img_width
		img = F.pad(img, (0,padSize,0,0,0,0), 'constant', 0)
		segm = np.pad(segm, ((0,0), (0, padSize)), 'constant', constant_values=255)
		cropBeginIdx_w = 0
	    else:
		# crop
		cropBeginIdx_w = random.randint(0, img_width-self.cropSize)
	    img = img[:, cropBeginIdx_h:(cropBeginIdx_h+self.cropSize), cropBeginIdx_w:(cropBeginIdx_w+self.cropSize)]
	    segm = segm[cropBeginIdx_h:(cropBeginIdx_h+self.cropSize), cropBeginIdx_w:(cropBeginIdx_w+self.cropSize)]

            batch_images[i][:, :img.shape[1], :img.shape[2]] = img
            batch_segms[i][:segm.shape[0], :segm.shape[1]] = torch.from_numpy(segm.astype(np.int)).long()

        output = dict()
        output['data'] = batch_images
        output['seg_label'] = batch_segms
        return output

    def __len__(self):
        return int(1e6) # It's a fake length due to the trick that every loader maintains its own list
        #return self.num_sampleclass


class VOCValDataset(torchdata.Dataset):
    def __init__(self, opt, max_sample=-1, start_idx=-1, end_idx=-1, val_batch_size=1):
        self.root_dataset = opt.root_dataset
        self.cropSize = opt.cropSize
	self.val_batch_size = val_batch_size

	# mean and std, in BGR order
        self.img_transform = transforms.Compose([ transforms.Normalize(mean=[104.008, 116.669, 122.675], std=[1., 1., 1.]) ])

        # load img and seg list
        self.list_sample = open(opt.val_list_file, 'r').readlines()

        self.cur_idx = 0

        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]

        if start_idx >= 0 and end_idx >= 0: # divide file list
            self.list_sample = self.list_sample[start_idx:end_idx]

        self.num_sample = len(self.list_sample)
        assert self.num_sample > 0
        print('# samples: {}'.format(self.num_sample))

    def _get_sub_batch(self):
        while True:
            # get a sample record
            this_sample = self.list_sample[self.cur_idx]
            self.batch_record_list.append(this_sample.strip())
            self.cur_idx += 1
	    # last batch of val data
            if self.cur_idx >= self.num_sample:
                batch_records = self.batch_record_list
                break
	    # finish a batch of val data
            if len(self.batch_record_list) == self.val_batch_size:
                batch_records = self.batch_record_list
                self.batch_record_list = []
                break
        return batch_records

    def __getitem__(self, index):
	# get sub-batch candidates
        batch_records = self._get_sub_batch()

	batch_images = torch.zeros(len(batch_records), 3, self.cropSize, self.cropSize)
        batch_segms = torch.zeros(len(batch_records), self.cropSize, self.cropSize).long()

	for i in range(len(batch_records)):
	    this_record = batch_records[i]

	    # 1 load image and label
	    image_path = os.path.normpath(self.root_dataset +  '/' + this_record.split(' ')[0])
            segm_path = os.path.normpath(self.root_dataset +  '/' + this_record.split(' ')[1])
            img = imread(image_path, mode='RGB')
            segm = imread(segm_path)
            assert(img.ndim == 3)
            assert(segm.ndim == 2)
            assert(img.shape[0] == segm.shape[0])
            assert(img.shape[1] == segm.shape[1])
	    # 2 RGB to BGR
            img = img.astype(np.float32)[:, :, ::-1] 
	    # 3 transpose from HxWxC to CxHxW
            img = img.transpose(2,0,1)
	    # 4 substract mean and normalize std
            img = self.img_transform(torch.from_numpy(img.copy()))
	    # 5 pad images to a fixed size
	    [img_height, img_width, _] = img.shape
	    assert(self.cropSize >= img_height)		# cropSize needs to be larger than image size
	    assert(self.cropSize >= img_width)
	    h_padSize = self.cropSize - img_height
	    w_padSize = self.cropSize - img_width
	    img = F.pad(img, (0,w_padSize,0,h_padSize,0,0), 'constant', 0)
	    segm = np.pad(segm, ((0, h_padSize), (0, w_padSize)), 'constant', constant_values=255)

	    batch_images[i][...] = img
            batch_segms[i][...] = torch.from_numpy(segm.astype(np.int)).long()

        output = dict()
        output['data'] = batch_images
        output['seg_label'] = batch_segms
        return output

    def __len__(self):
        return self.num_sample


class VOCTestDataset(torchdata.Dataset):
    def __init__(self, opt, max_sample=-1):
        self.root_dataset = opt.root_dataset
        self.cropSize = opt.cropSize

        # mean and std, in BGR order
        self.img_transform = transforms.Compose([ transforms.Normalize(mean=[104.008, 116.669, 122.675], std=[1., 1., 1.]) ])

        # load img and seg list
        self.list_sample = open(opt.test_list_file, 'r').readlines()

        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]
        self.num_sample = len(self.list_sample)
        assert self.num_sample > 0
        print('# samples: {}'.format(self.num_sample))

    def __getitem__(self, index):
        this_record = self.list_sample[index]
        # load and preprocess image
        image_path = os.path.normpath(self.root_dataset +  '/' + this_record.strip())
        img = imread(image_path, mode='RGB')
	assert(img.ndim == 3)
        img = img.astype(np.float32)[:, :, ::-1] 
	img = img.transpose(2,0,1)
	img = self.img_transform(torch.from_numpy(img.copy()))

        output = dict()
        output['data'] = img
        return output

    def __len__(self):
        return self.num_sample
