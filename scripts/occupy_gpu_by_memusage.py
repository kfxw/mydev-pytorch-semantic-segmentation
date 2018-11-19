import pynvml

def a():   
	b=0
	while True:
		handle = pynvml.nvmlDeviceGetHandleByIndex(4)
		meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
		if meminfo.free < 1024*1024*1024*23:
			b=4
		else:
			break
		handle = pynvml.nvmlDeviceGetHandleByIndex(5)
		meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
		if meminfo.free < 1024*1024*1024*23:
			b=5
		else:
			break
		handle = pynvml.nvmlDeviceGetHandleByIndex(6)
		meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
		if meminfo.free < 1024*1024*1024*23:
			b=6
			print '!!!'
			time.sleep(10)
		else:
			break
	os.system('/home/kfxw/Development/caffe-1612/build/tools/caffe train -solver ./ade20k/solver_vgg.prototxt -weights ./vgg16_20M.caffemodel -gpu {} 2>&1 | tee vgg_ade20k_dilation3.log'.format(b))


