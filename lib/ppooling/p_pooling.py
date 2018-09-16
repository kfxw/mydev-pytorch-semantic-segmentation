import pdb
import cupy
import torch
import math
import torch.nn.functional as F

@cupy.util.memoize(for_each_device=True)
def cunnex(strFunction):
	return cupy.cuda.compile_with_cache(globals()[strFunction]).get_function(strFunction)


DensePNormForward = """
extern "C" __global__ void DensePNormForward(const int nthreads,
	 const float* padded_bottom_data, float* top_data,
	 const float* p_data, double* numerator_data, double* denominator_data,
	 const int bottom_num, const int channels,
	 const int padded_bottom_height_, const int padded_bottom_width_,
	 const int pooled_height_, const int pooled_width_,
	 const int kernel_h_, const int kernel_w_,
	 const int stride_h_, const int stride_w_,
	 const int pad_h_, const int pad_w_) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= nthreads)  return;
    const int pw = index % pooled_width_;
    const int ph = (index / pooled_width_) % pooled_height_;
    const int c = (index / pooled_width_ / pooled_height_) % channels;
    const int n = index / pooled_width_ / pooled_height_ / channels;
    int hstart = ph * stride_h_;
    int wstart = pw * stride_w_;
    int hend = min(hstart + kernel_h_, padded_bottom_height_);
    int wend = min(wstart + kernel_w_, padded_bottom_width_);
    double tmp_numerator = 0;
    double tmp_denominator = double(1e-20);
    int top_idx = index;
    padded_bottom_data += (n * channels + c) * padded_bottom_height_ * padded_bottom_width_;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
	int bottom_idx = h * padded_bottom_width_ + w;

	double x_pow_p = 0;
	double a = (double)padded_bottom_data[bottom_idx];
	double b = (double)p_data[top_idx];
	if (abs(b - 1) < 1e-34)
	  x_pow_p = a;
	else {
	  int e = (int) b;
	  union {	double d;	int x[2];    }	 u = { a };
	  u.x[1] = (int)((b - e) * (u.x[1] - 1072632447) + 1072632447);
	  u.x[0] = 0;
	  double r = 1.0;
	    while (e) {
	    if (e & 1) { r *= a; }
	    a *= a;
	    e >>= 1;
	  }
	  x_pow_p = r * u.d;
	}

	double x_pow_p_plus1 = x_pow_p * padded_bottom_data[bottom_idx];
	tmp_numerator += x_pow_p_plus1;
	tmp_denominator += x_pow_p;
      }
    }
    top_data[top_idx] = (float)(tmp_numerator / tmp_denominator);
    numerator_data[top_idx] = tmp_numerator;
    denominator_data[top_idx] = tmp_denominator;
}
"""

DensePNormBackward_P = """
extern "C" __global__ void DensePNormBackward_P(const int nthreads,
	 const float* padded_bottom_data, const float* top_data, const float* top_diff,
	 const float* p_data, float* p_diff,
	 const double* numerator_data, const double* denominator_data,
	 const int bottom_num, const int channels,
	 const int padded_bottom_height_, const int padded_bottom_width_,
	 const int pooled_height_, const int pooled_width_,
	 const int kernel_h_, const int kernel_w_,
	 const int stride_h_, const int stride_w_,
	 const int pad_h_, const int pad_w_) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= nthreads)  return;
    const int pw = index % pooled_width_;
    const int ph = (index / pooled_width_) % pooled_height_;
    const int c = (index / pooled_width_ / pooled_height_) % channels;
    const int n = index / pooled_width_ / pooled_height_ / channels;
    int hstart = ph * stride_h_;
    int wstart = pw * stride_w_;
    int hend = min(hstart + kernel_h_, padded_bottom_height_);
    int wend = min(wstart + kernel_w_, padded_bottom_width_);
    int top_idx = index;
    double sum1 = 0.0;	
    double sum2 = 0.0;	
    int bottom_offset = (n * channels + c) * padded_bottom_height_ * padded_bottom_width_;
    padded_bottom_data += bottom_offset;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
	int bottom_idx = h * padded_bottom_width_ + w;

	double x_pow_p = 0;
	double a = (double)padded_bottom_data[bottom_idx];
	double b = (double)p_data[top_idx];
	if (abs(b - 1) < 1e-34)
	  x_pow_p = a;
	else {
	  int e = (int) b;
	  union {	double d;	int x[2];    }	 u = { a };
	  u.x[1] = (int)((b - e) * (u.x[1] - 1072632447) + 1072632447);
	  u.x[0] = 0;
	  double r = 1.0;
	    while (e) {
	    if (e & 1) { r *= a; }
	    a *= a;
	    e >>= 1;
	  }
	  x_pow_p = r * u.d;
	}

	double x_pow_p_plus1 = x_pow_p * (double)padded_bottom_data[bottom_idx];
	float bottom_data_value = padded_bottom_data[bottom_idx]<1e-3 ? 1e-3 : padded_bottom_data[bottom_idx];
	bottom_data_value = log(bottom_data_value);
	sum1 += bottom_data_value * x_pow_p_plus1;
	sum2 += bottom_data_value * x_pow_p;
      }
    }
    float tmp = (float) ((sum1 - sum2*top_data[top_idx]) / (denominator_data[top_idx]+1e-10));
    p_diff[top_idx] = top_diff[top_idx] * (float)tmp;
}
"""

DensePNormBackward_data = """
extern "C" __global__ void DensePNormBackward_data(const int nthreads,
	 const float* padded_bottom_data, float* bottom_diff, const float* top_data, const float* top_diff,
	 const float* p_data,
	 const double* numerator_data, const double* denominator_data,
	 const int bottom_num, const int channels,
	 const int bottom_height_, const int bottom_width_,
	 const int padded_bottom_height_, const int padded_bottom_width_,
	 const int pooled_height_, const int pooled_width_,
	 const int kernel_h_, const int kernel_w_,
	 const int stride_h_, const int stride_w_,
	 const int pad_h_, const int pad_w_) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= nthreads)  return;
    const int w = index % padded_bottom_width_;
    const int h = (index / padded_bottom_width_) % padded_bottom_height_;
    const int c = (index / padded_bottom_width_ / padded_bottom_height_) % channels;
    const int n = index / padded_bottom_width_ / padded_bottom_height_ / channels;
    const int phstart = (h < kernel_h_) ? 0 : (h - kernel_h_) / stride_h_ + 1;
    const int phend = min(h / stride_h_ + 1, pooled_height_);
    const int pwstart = (w < kernel_w_) ? 0 : (w - kernel_w_) / stride_w_ + 1;
    const int pwend = min(w / stride_w_ + 1, pooled_width_);
    int bottom_idx = index;
    bottom_diff += (n * channels + c) * bottom_height_ * bottom_width_;
    int top_offset = (n * channels + c) * pooled_height_ * pooled_width_;
    top_data += top_offset;
    top_diff += top_offset;
    p_data += top_offset;
    numerator_data += top_offset;
    denominator_data += top_offset;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
	if ((h >= pad_h_) && (w >= pad_w_) && (h < bottom_height_ + pad_h_) && (w < bottom_width_ + pad_w_))
	{
		int top_idx = ph * pooled_width_ + pw;

		double x_pow_p_minus1 = 0;
		double a = (double)padded_bottom_data[bottom_idx]+1e-10;
		double b = (double)p_data[top_idx]-1;
		if (abs(b - 1) < 1e-34)
		  x_pow_p_minus1 = a;
		else {
		  int e = (int) b;
		  union {	double d;	int x[2];    }	 u = { a };
		  u.x[1] = (int)((b - e) * (u.x[1] - 1072632447) + 1072632447);
		  u.x[0] = 0;
		  double r = 1.0;
		    while (e) {
		    if (e & 1) { r *= a; }
		    a *= a;
		    e >>= 1;
		  }
		  x_pow_p_minus1 = r * u.d;
		}

		double x_pow_p = 0;
		a = (double)padded_bottom_data[bottom_idx];
		b = (double)p_data[top_idx];
		if (abs(b - 1) < 1e-34)
		  x_pow_p = a;
		else {
		  int e = (int) b;
		  union {	double d;	int x[2];    }	 u = { a };
		  u.x[1] = (int)((b - e) * (u.x[1] - 1072632447) + 1072632447);
		  u.x[0] = 0;
		  double r = 1.0;
		    while (e) {
		    if (e & 1) { r *= a; }
		    a *= a;
		    e >>= 1;
		  }
		  x_pow_p = r * u.d;
		}

		int ori_bottom_idx = (h-pad_h_) * bottom_width_ + (w-pad_w_);
		float tmp = ((p_data[top_idx]+1) * x_pow_p - p_data[top_idx] * x_pow_p_minus1 * top_data[top_idx]) / (denominator_data[top_idx]+1e-10);
		bottom_diff[ori_bottom_idx] += top_diff[top_idx] * (float)tmp;
	}
      }
    }
}
"""

class P_Pooling(torch.autograd.Function):
	def __init__(self, kernel_size, padding = 0, stride = 1):
		super(P_Pooling, self).__init__()
		self.kernel_size = kernel_size
		self.pad = padding
		self.stride = stride

	def forward(self, bottom_0, bottom_1):

		assert(bottom_0.is_contiguous() == True)
		assert(bottom_1.is_contiguous() == True)
		assert(bottom_0.size()[2]==bottom_0.size()[3])	# only handle square feature maps
		assert(bottom_1.size()[2]==bottom_1.size()[3])

		# shift bottom_0 data to non-negative space
		bottom_0_min = bottom_0.min(keepdim=True,dim=3)[0].min(keepdim=True,dim=2)[0]		# find min values of each channel
		bottom_0_min = -(F.threshold(-bottom_0_min, 0, 0, inplace=False))			# ignore positive mins
		#bottom_0_min.detach()
		bottom_0 = bottom_0 - bottom_0_min							# shift input data

		num = bottom_0.size()[0]
		channels = bottom_0.size()[1]
		bottom_size = bottom_0.size()[2]

		# reshape top
		#pooled_size = (int)(math.ceil((float)(bottom_size + 2 * self.pad - self.kernel_size) / self.stride)) + 1
		pooled_size = (int)(math.floor((float)(bottom_size + 2 * self.pad - (self.kernel_size - 1) - 1) / self.stride) + 1)
		if self.pad != 0 :
			if (pooled_size - 1) * self.stride >= bottom_size + self.pad :
				pooled_size -= 1
			assert(((pooled_size - 1) * self.stride) < (bottom_size + self.pad));
		top = bottom_0.new_zeros((num, channels, pooled_size, pooled_size))
		assert(pooled_size==bottom_1.size()[2]), "{} vs. {}".format(pooled_size, bottom_1.size()[2])

		# preprocess
		# init numerator and denominator
		numerator_data = top.new_zeros(top.size(), dtype=torch.float64);
		denominator_data = top.new_zeros(top.size(), dtype=torch.float64);
		# pad bottom[0]
		padded_input = F.pad(bottom_0, (self.pad, self.pad, self.pad, self.pad, 0, 0, 0, 0), mode='constant', value=0)
		padded_input_size = padded_input.size()[2]

		# forward
		if bottom_0.is_cuda == True and bottom_1.is_cuda == True:
			class Stream:
				ptr = torch.cuda.current_stream().cuda_stream
			count = top.nelement()
			cunnex('DensePNormForward')(
				grid=tuple([ int((count + 512 - 1) / 512), 1, 1 ]),
				block=tuple([ 512, 1, 1 ]),
				args=[ count, padded_input.data_ptr(), top.data_ptr(),\
				       bottom_1.data_ptr(), numerator_data.data_ptr(), denominator_data.data_ptr(),\
				       num, channels, padded_input_size, padded_input_size, pooled_size, pooled_size,\
				       self.kernel_size, self.kernel_size, self.stride, self.stride, self.pad, self.pad ],
				stream=Stream
			)
			self.save_for_backward(bottom_1, padded_input, top, numerator_data, denominator_data)
		else:
			raise NotImplementedError()

		return top + bottom_0_min

	def backward(self, top_diff):
		bottom_1, padded_input, top, numerator_data, denominator_data = self.saved_tensors
		num = padded_input.size()[0]
		channels = padded_input.size()[1]
		padded_input_size = padded_input.size()[2]
		pooled_size = top.size()[2]

		assert(top_diff.is_contiguous() == True)

		bottom_size = padded_input_size - 2*self.pad
		bottom_0_diff = padded_input.new_zeros((num, channels, bottom_size, bottom_size))
		bottom_1_diff = bottom_1.new_zeros(bottom_1.size())

		if bottom_1.is_cuda == True:
			class Stream:
				ptr = torch.cuda.current_stream().cuda_stream

			top_count = top_diff.nelement()
			padded_input_count = padded_input.nelement()
			# gradient w.r.t. p
			cunnex('DensePNormBackward_P')(
				grid=tuple([ int((top_count + 512 - 1) / 512), 1, 1 ]),
				block=tuple([ 512, 1, 1 ]),
				args=[ top_count, padded_input.data_ptr(), top.data_ptr(), top_diff.data_ptr(),\
				       bottom_1.data_ptr(), bottom_1_diff.data_ptr(),\
				       numerator_data.data_ptr(), denominator_data.data_ptr(),\
				       num, channels, padded_input_size, padded_input_size,\
				       pooled_size, pooled_size,\
				       self.kernel_size, self.kernel_size, self.stride, self.stride, self.pad, self.pad ],
				stream=Stream
			)
			# gradient w.r.t. data
			cunnex('DensePNormBackward_data')(
				grid=tuple([ int((padded_input_count + 512 - 1) / 512), 1, 1 ]),
				block=tuple([ 512, 1, 1 ]),
				args=[ padded_input_count, padded_input.data_ptr(), bottom_0_diff.data_ptr(), top.data_ptr(), top_diff.data_ptr(), bottom_1.data_ptr(),\
				       numerator_data.data_ptr(), denominator_data.data_ptr(),\
				       num, channels, bottom_size, bottom_size,\
				       padded_input_size, padded_input_size, pooled_size, pooled_size,\
				       self.kernel_size, self.kernel_size, self.stride, self.stride, self.pad, self.pad ],
				stream=Stream
			)
		else:
			raise NotImplementedError()

		return bottom_0_diff, bottom_1_diff
