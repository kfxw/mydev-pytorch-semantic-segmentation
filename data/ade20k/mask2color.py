import numpy as np
import scipy.io as sio

colormap = sio.loadmat('./data/ade20k/color150.mat')['color']

def Mask2ColorMap(mask):
	color = np.zeros((mask.shape[0], mask.shape[1], 3))
	color[mask==0,: = [255,255,255]
	for i in range(150):
		color[mask==i+1, :] = colormap[i,:]

	return color
