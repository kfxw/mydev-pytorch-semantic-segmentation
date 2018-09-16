class opt():
	def __init__(self):
	## network settings
		# [str], the name of net
		self.id = 'deeplabv3_xception_allbn_pnorm'
		# [str], arch names for network encoder and decoder
		#    see networks/ModelBuilder.py for decoder definitions and
		#    the other files for encoder definitions
		self.arch_encoder = 'xception'
		self.arch_decoder = 'deeplabv3_aspp_bilinear'	# 'deeplabv3_aspp_bilinear_deepsup'
		# [str/None], pretrained models to initialize the network
		#    if None, the corresponding layers are randomly initialized
		self.weights_encoder = 'xception.pth'#'trainingResults/deeplabv3_xception_allbn_pnorm_xception_deeplabv3_aspp_bilinear_VOC2012aug_ngpus1_batchSize24_trainCropSize504_LR_encoder1e-07_LR_decoder1e-07_epoch46+0912-10:25:47/encoder_epoch_46.pth'
		self.weights_decoder = ''#'trainingResults/deeplabv3_xception_allbn_pnorm_xception_deeplabv3_aspp_bilinear_VOC2012aug_ngpus1_batchSize24_trainCropSize504_LR_encoder1e-07_LR_decoder1e-07_epoch46+0912-10:25:47/decoder_epoch_46.pth'
		# [double/None], the weight of deep supervision loss
		self.deep_sup_scale = None
		# [int: 8,16,32], decide the overall stride (strides in the last two network stages)
		self.overall_stride = 16

	## training settings
		# [int], seed for random generators
		self.seed = 888

		## device related
		# [int list], id of used gpus, begins from 0
		self.gpu_id = [0]
		# automatically obtained
		self.num_gpus = len(self.gpu_id)
		# [int], total batch size = num_gpu x batch_size_per_gpu
		self.batch_size_per_gpu = 16

		## optim related
		# [str], optimizer, currently SGD only
		self.optim = 'SGD'
		# [double], base learning rates, momentum, weight decay
		self.lr_encoder = 1e-7#2e-8##1.5e-9 #
		self.lr_decoder = 1e-7#2e-8##1.5e-9 #
		self.momentum = 0.9
		self.weight_decay = 5e-4
		# [double], the power used in lr poly stepping
		self.lr_pow = 0.9
		# [int], max number of iteration epochs (begins from 1)
		self.num_epoch = 46
		# [int], the number of beginning epoch when resuming from a snapshot (begins from 1)
		self.start_epoch = 1
		# [int], iterations of each epoch (irrelevant to batch size)
		self.epoch_iters = 661
		# [str], the snapshot are saved at snapshot_prefix+id folder
		self.snapshot_prefix = './trainingResults/'

		## test settings
		# [int], test the model for every test_epoch_interval epochs
		self.test_epoch_interval = 1
		# [int], batch size of validation
		self.val_batch_size = 4

	## dataset settings
		# [str], dataset name, for display only
		self.dataset_name = 'VOC2012aug'
		# [int], number of output classes
		self.num_class = 21
		# [str], root path of dataset 
		self.root_dataset = '/data/zhenwei/Development/data/VOC/VOC2012-May-11-data/VOCdevkit/VOC2012'
		# [str], file that contain image/gt file names (2 columns divied by a space)
		self.train_list_file = '/data/zhenwei/Development/data/VOC/VOC2012-May-11-data/VOCdevkit/VOC2012/ImageSets/Segmentation/official_trainval.txt'#/deeplab_train_aug.txt
		self.val_list_file = '/data/zhenwei/Development/data/VOC/VOC2012-May-11-data/VOCdevkit/VOC2012/ImageSets/Segmentation/deeplab_val.txt'
		# [str], file that contain image file names (1 columns)
		self.test_list_file = '/data/zhenwei/Development/data/VOC/VOC2012-May-11-data/VOCdevkit/VOC2012/ImageSets/Segmentation/deeplab_test.txt'#deeplab_val_no_gt.txt
		# [int], image size during training and testing
		#   during training, images are cropped (or padded to) cropSize
		#   during testing, cropSize must be larger than image sizes and images are padded
		self.train_cropSize = 504	# 512,328
		self.valtest_cropSize = 504
		# [bool], whether to conduct random horizontal flip during training
		self.random_flip = True
		# [bool], whther to apply random resizing during training
		self.random_scale = True
		# [double], if random_sace is True, belows decide the range of resizing factor
		self.random_scale_factor_max = 2
		self.random_scale_factor_min = 0.5

	## display settings
		# [int], the interval of printing the loss during traning
		self.display_interval = 1
