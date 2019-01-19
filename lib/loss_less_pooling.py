import pdb
import cupy
import torch
import math
import torch.nn.functional as F
import numpy as np

@cupy.util.memoize(for_each_device=True)
def cunnex(strFunction):
	return cupy.cuda.compile_with_cache(globals()[strFunction]).get_function(strFunction)


LossLessPoolForward = """
extern "C" __global__ void LossLessPoolForward(const int nthreads,
	 const float* bottom_data, float* top_data, const int rate,
	 const int num, const int channels,
	 const int bottom_height_, const int bottom_width_,
	 const int pooled_height_, const int pooled_width_) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= nthreads)  return;
    const int rate_p2 = rate * rate;
    const int pw = index % pooled_width_;
    const int ph = (index / pooled_width_) % pooled_height_;
    const int c = (index / pooled_width_ / pooled_height_) % channels;
    const int rate_idx = (index / pooled_width_ / pooled_height_/ channels) % rate_p2;
    const int n = index / pooled_width_ / pooled_height_ / channels / rate_p2;

    int rate_idx_w = rate_idx % rate;		// offset in one pooling window
    int rate_idx_h = rate_idx / rate;

    int bottom_offset = (n * channels + c) * bottom_height_ * bottom_width_;
    int bottom_idx_h = min(ph * rate + rate_idx_h, bottom_height_);
    int bottom_idx_w = min(pw * rate + rate_idx_w, bottom_width_);

    top_data[index] = bottom_data[bottom_offset + bottom_idx_h*bottom_width_ + bottom_idx_w];
}
"""

LossLessPoolBackward = """
extern "C" __global__ void LossLessPoolBackward(const int nthreads,
	 float* bottom_diff, const float* top_diff, const int rate,
	 const int num, const int channels,
	 const int bottom_height_, const int bottom_width_,
	 const int pooled_height_, const int pooled_width_) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= nthreads)  return;
    const int w = index % bottom_width_;
    const int h = (index / bottom_width_) % bottom_height_;
    const int c = (index / bottom_width_ / bottom_height_) % channels;
    const int n = index / bottom_width_ / bottom_height_ / channels;
    int pw = w / rate;
    int ph = h / rate;
    if ((ph >= pooled_height_) || (pw >= pooled_width_)) {
	bottom_diff[index] = 0;
	return;
    }

    int rate_idx_w = w % rate;
    int rate_idx_h = h % rate;
    int rate_idx = rate_idx_h * rate + rate_idx_w;

    const int rate_p2 = rate * rate;
    int top_offset = ((n * rate_p2 + rate_idx) * channels + c) * pooled_height_ * pooled_width_;
    int top_idx_h = ph;
    int top_idx_w = pw;
    bottom_diff[index] = top_diff[top_offset + top_idx_h*pooled_width_ + top_idx_w];
}
"""

class Loss_Less_Pooling(torch.autograd.Function):
	def __init__(self, down_sampling_rate):
		super(Loss_Less_Pooling, self).__init__()
		self.rate = down_sampling_rate

		assert isinstance(self.rate, int), 'Only support Int down sampling rate.'
		assert (self.rate > 1) and (self.rate <= 5), 'Down sampling rate must be the one of [2,3,4,5].'

		self.save = []

	def forward(self, bottom_0):
		assert bottom_0.is_contiguous() == True, 'Input feature is not contiguous.'
		assert bottom_0.size()[2]==bottom_0.size()[3], 'Only handle square feature maps.'	# only handle square feature maps

		num, channels, bottom_size, _ = bottom_0.size()

		# reshape top
		pooled_size = (int)(math.floor((float)(bottom_size) / self.rate))
		top = bottom_0.new_zeros((num, self.rate*self.rate, channels, pooled_size, pooled_size), dtype=torch.float32)

		# forward
		if bottom_0.is_cuda == True:
			class Stream:
				ptr = torch.cuda.current_stream().cuda_stream
			count = top.nelement()
			cunnex('LossLessPoolForward')(
				grid=tuple([ int((count + 512 - 1) / 512), 1, 1 ]),
				block=tuple([ 512, 1, 1 ]),
				args=[ count, bottom_0.data_ptr(), top.data_ptr(), self.rate,\
				       num, channels, bottom_size, bottom_size, pooled_size, pooled_size ],
				stream=Stream
			)

			self.save.append(bottom_size)
		else:
			raise NotImplementedError()
		return top

	def backward(self, top_diff):
		num, __, channels, pooled_size, _ = top_diff.size()
		bottom_size = self.save[0]
		self.save = []

		assert top_diff.is_contiguous() == True, 'Top diff is not contiguous.'
		bottom_0_diff = top_diff.new_zeros((num, channels, bottom_size, bottom_size), dtype=torch.float32)

		if top_diff.is_cuda == True:
			class Stream:
				ptr = torch.cuda.current_stream().cuda_stream

			bottom_count = bottom_0_diff.nelement()
			# gradient w.r.t. data
			cunnex('LossLessPoolBackward')(
				grid=tuple([ int((bottom_count + 512 - 1) / 512), 1, 1 ]),
				block=tuple([ 512, 1, 1 ]),
				args=[ bottom_count, bottom_0_diff.data_ptr(), top_diff.data_ptr(), self.rate,\
				       num, channels, bottom_size, bottom_size, pooled_size, pooled_size ],
				stream=Stream
			)
		else:
			raise NotImplementedError()

		return bottom_0_diff


# test code
# generate input
in_ = np.array([1,2,3,1,2,3,1,\
		1,2,3,1,2,3,1,\
		1,2,3,1,2,3,1,\
		1,2,3,1,2,3,1,\
		1,2,3,1,2,3,1,\
		1,2,3,1,2,3,1,\
		1,2,3,1,2,3,1]).reshape(1,1,7,7).astype(np.float32)
in_ = np.tile(in_, [1,2,1,1])
in_[:,1,:,:] -= 1
# create loss less pooling (N,C,H,W) -> (N,stride^2,C,H/stride,W/stride)
stride = 2
a = Loss_Less_Pooling(stride)
res = a(torch.from_numpy(in_).cuda())
print res
# reverse loss less pooling by DUC (pixel shuffle), with padding to original size
b = torch.nn.PixelShuffle(stride)
pad_size = in_.shape[-1] % stride
print torch.nn.functional.pad(b(torch.transpose(res,1,2).contiguous().view(1,18,2,2)), (0,pad_size,0,pad_size,0,0,0,0))

# generate diff
in_ = np.array([1,1,1,1,\
		2,2,2,2,\
		3,3,3,3,\
		4,4,4,4,\
		5,5,5,5,\
		6,6,6,6,\
		7,7,7,7,\
		8,8,8,8,\
		9,9,9,9]).reshape(1,9,1,2,2).astype(np.float32)
in_ = np.tile(in_, [1,1,2,1,1])
in_[:,:,1,:,:] *= -1
# loss less pooling backward
res = a.backward(torch.from_numpy(in_).cuda())
print in_
print res
