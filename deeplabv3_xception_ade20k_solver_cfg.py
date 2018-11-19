class opt():
	def __init__(self):
	## network settings
		# [str], the name of net
		self.id = 'deeplabv3_xception_8s_allbn_lr9-270-90-x10_scale10_pnorm_initp50max90'
		#'deeplabv3_xception_allbn_8s' #'debug'
		# [str], arch names for network encoder and decoder
		#    see networks/ModelBuilder.py for decoder definitions and
		#    the other files for encoder definitions
		self.arch_encoder = 'xception'
		self.arch_decoder = 'deeplabv3_aspp_bilinear'	# 'deeplabv3_aspp_bilinear_deepsup'
		# [str/None], pretrained models to initialize the network
		#    if None, the corresponding layers are randomly initialized
		self.weights_encoder = 'trainingResults/deeplabv3_xception_allbn_lr3-30-100-x3_scale10_pnorm_initp50max90_xception_deeplabv3_aspp_bilinear_ade20k_ngpus1_batchSize8_trainCropSize600_LR_encoder1e-07_LR_decoder1e-07_epoch35+1020-23:59:41/encoder_epoch_35.pth'
		#'trainingResults/deeplabv3_xception_allbn_xception_deeplabv3_aspp_bilinear_ade20k_ngpus2_batchSize8_trainCropSize600_LR_encoder1e-07_LR_decoder1e-07_epoch35+0926-16%3A56%3A18/encoder_epoch_35.pth' #'xception.pth'
		self.weights_decoder = 'trainingResults/deeplabv3_xception_allbn_lr3-30-100-x3_scale10_pnorm_initp50max90_xception_deeplabv3_aspp_bilinear_ade20k_ngpus1_batchSize8_trainCropSize600_LR_encoder1e-07_LR_decoder1e-07_epoch35+1020-23:59:41/decoder_epoch_35.pth'
		# 'trainingResults/deeplabv3_xception_allbn_xception_deeplabv3_aspp_bilinear_ade20k_ngpus2_batchSize8_trainCropSize600_LR_encoder1e-07_LR_decoder1e-07_epoch35+0926-16%3A56%3A18/decoder_epoch_35.pth'#''
		# [double/None], the weight of deep supervision loss
		self.deep_sup_scale = None
		# [int: 8,16,32], decide the overall stride (strides in the last two network stages)
		self.overall_stride = 8

	## training settings
		# [int], seed for random generators
		self.seed = 888

		## device related
		# [int list], id of used gpus, begins from 0
		self.gpu_id = [0]
		# automatically obtained
		self.num_gpus = len(self.gpu_id)
		# [int], total batch size = num_gpu x batch_size_per_gpu
		self.batch_size_per_gpu = 4

		## optim related
		# [str], optimizer, currently SGD only
		self.optim = 'SGD'
		# [double], base learning rates, momentum, weight decay
		self.lr_encoder = 1e-8#2e-8#1e-7#
		self.lr_decoder = 1e-8#2e-8#1e-7#
		self.momentum = 0.9
		self.weight_decay = 5e-4
		# [double], the power used in lr poly stepping
		self.lr_pow = 0.9
		# [int], max number of iteration epochs (begins from 1)
		self.num_epoch = 60
		# [int], the number of beginning epoch when resuming from a snapshot (begins from 1)
		self.start_epoch = 1
		# [int], iterations of each epoch (irrelevant to batch size)
		self.epoch_iters = 2221
		# [str], the snapshot are saved at snapshot_prefix+id folder
		self.snapshot_prefix = './trainingResults/'

		## test settings
		# [int], test the model for every test_epoch_interval epochs
		self.test_epoch_interval = 2
		# [int], batch size of validation
		self.val_batch_size = 4

	## dataset settings
		# [str], dataset name, for display only
		self.dataset_name = 'ade20k'
		# [int], number of output classes
		self.num_class = 151
		# [str], root path of dataset 
		self.root_dataset = '/home/kfxw/Development/data/ADE20K/'
		# [str], file that contain image/gt file names (2 columns divied by a space)
		self.train_list_file = '/home/kfxw/Development/data/ADE20K/trainval_list.txt'
		self.val_list_file = '/home/kfxw/Development/data/ADE20K/val_smaller_than_896_list.txt'
		# [str], file that contain image file names (1 columns)
		self.test_list_file = '/home/kfxw/Development/data/ADE20K/test_list.txt'
		# [int], image size during training and testing
		#   during training, images are cropped (or padded to) cropSize
		#   during testing, cropSize must be larger than image sizes and images are padded
		self.train_cropSize = 600	# 512,328
		self.valtest_cropSize = 896
		# [bool], whether to conduct random horizontal flip during training
		self.random_flip = True
		# [bool], whther to apply random resizing during training
		self.random_scale = True
		# [double], if random_sace is True, belows decide the range of resizing factor
		self.random_scale_factor_max = 2
		self.random_scale_factor_min = 0.5

	## display settings
		# [int], the interval of printing the loss during traning
		self.display_interval = 20
