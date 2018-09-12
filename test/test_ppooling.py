import numpy as np
import torch
import sys,os
sys.path.append('../')
from lib import ppooling
import torch.nn.functional as F

class Network(torch.nn.Module):
	def __init__(self):
		super(Network, self).__init__()
		self.pooling = ppooling.P_Pooling(2,padding=1)
		
	# end

	def forward(self, input1, input2):
		return self.pooling(input1, input2) #F.avg_pool2d(input1, 2,padding=1)

net = Network().cuda()
net.zero_grad()

# original caffe test
"""
data = np.array([0.001,1200,0,0,0,10,0,0]).reshape(2,1,2,2).astype(float)
p_map = np.zeros([2,1,4,4]).astype(float)
p_map[...] = 10

net.blobs['data'].data[...] = data
net.blobs['p_map'].data[...] = p_map
net.forward()

# Forward, padding 1
# data:    p:    |  results:
# 1  2     2  2  |  1   1.5   2
# 3  4     2  2  | 2.8  3.33  3.6
#                |  3   3.64  4
print 'Forward-----------------'
print net.blobs['data'].data
print '------------------------'
print net.blobs['p_norm'].data
print '------------------------'

net.blobs['p_norm'].diff[...] = 1
net.backward()

# Backward
# top diff:  data diff:         p diff:
# 1 1 1      0.3900  1.5614    0.0000  0.1109   0.0000
# 1 1 1      2.3412  3.6076    0.1978  0.2598   0.2218
# 1 1 1                        0.0000  0.0663   0.0000
print 'Backward----------------'
print net.blobs['data'].diff
print '------------------------'
print net.blobs['p_map'].diff
print '------------------------'
"""

data = torch.from_numpy(np.array([1,2,3,4]).reshape(1,1,2,2).astype(float)).cuda()-2
data = torch.from_numpy(np.array([0,1,2,3]).reshape(1,1,2,2).astype(float)).cuda()
p_map = torch.from_numpy(np.zeros([1,1,3,3]).astype(float)).cuda()
p_map[...] = 2

data.requires_grad_()
p_map.requires_grad_()

res = net(data,p_map)

print data
print p_map
print res

diff = torch.from_numpy(np.zeros([1,1,3,3]).astype(float)).cuda()
diff[...] = 1
res.backward(gradient=diff)

print data.grad
print p_map.grad
#print torch.autograd.gradcheck(net, tuple([ data, p_map ]), 0.001)
