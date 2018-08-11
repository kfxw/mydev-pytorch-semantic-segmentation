class opt():
	def __init__(self):
	## network settings
		# [str], the name of net
		self.id = 'test'
		# [str], arch names for network encoder and decoder
		#    see networks/ModelBuilder.py for decoder definitions and
		#    the other files for encoder definitions
		self.arch_encoder = 'resnet50_dilated8'
		self.arch_decoder = 'ppm_bilinear_deepsup'
		# [str/None], pretrained models to initialize the network
		#    if None, the corresponding layers are randomly initialized
		self.weights_encoder = ''
		self.weights_decoder = ''
		# [double/None], the weight of deep supervision loss
		self.deep_sup_scale = 0.4

	## training settings
		# [int], seed for random generators
		self.seed = 304

		## device related
		# [int list], id of used gpus
		self.gpu_id = [1]
		# automatically obtained
		self.num_gpus = len(self.gpu_id)
		# [int], total batch size = num_gpu x batch_size_per_gpu
		self.batch_size_per_gpu = 2

		## optim related
		# [str], optimizer, currently SGD only
		self.optim = 'SGD'
		# [double], base learning rates, momentum, weight decay
		self.lr_encoder = 2e-2
		self.lr_decoder = 2e-2
		self.momentum = 0.9
		self.weight_decay = 1e-4
		# [double], the power used in lr poly stepping
		self.lr_pow = 0.9
		# [int], max number of iteration epochs (begins from 1)
		self.num_epoch = 20
		# [int], the number of beginning epoch when resuming from a snapshot (begins from 1)
		self.start_epoch = 1
		# [int], iterations of each epoch (irrelevant to batch size)
		self.epoch_iters = 5000
		# [str], the snapshot are saved at snapshot_prefix+id folder
		self.snapshot_prefix = './trainingResults/'

	## dataset settings
		# [str], dataset name, for display only
		self.dataset_name = 'VOC2012aug'
		# [int], number of output classes
		self.num_class = 21
		# [str], root path of dataset 
		self.root_dataset = '/home/kfxw/Development/data/VOC/VOC2012-May-11-data/VOCdevkit/VOC2012'
		# [str], file that contain image/gt file names (2 columns divied by a space)
		self.train_list_file = '/home/kfxw/Development/data/VOC/VOC2012-May-11-data/VOCdevkit/VOC2012/ImageSets/Segmentation/deeplab_trainval_aug.txt'
		self.val_list_file = '/home/kfxw/Development/data/VOC/VOC2012-May-11-data/VOCdevkit/VOC2012/ImageSets/Segmentation/deeplab_val.txt'
		# [str], file that contain image file names (1 columns)
		self.test_list_file = '/home/kfxw/Development/data/VOC/VOC2012-May-11-data/VOCdevkit/VOC2012/ImageSets/Segmentation/deeplab_val.txt'
		# [int], image size during training and testing
		#   during training, images are cropped (or padded to) cropSize
		#   during testing, cropSize must be larger than image sizes and images are padded
		self.cropSize = 512	# 512,328
		# [bool], whether to conduct random horizontal flip during training
		self.random_flip = True
		# [bool], whther to apply random resizing during training
		self.random_scale = True
		# [double], if random_sace is True, belows decide the range of resizing factor
		self.random_scale_factor_max = 1.2
		self.random_scale_factor_min = 0.75

	## display settings
		# [int], the interval of printing the loss during traning
		self.display_interval = 20
