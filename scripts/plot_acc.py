import numpy as np
import re
import matplotlib.pyplot as plt

p = re.compile("Mean IoU: 0.[0-9]+")
s = file('./deeplabv3_xception_allbn_trainset_only_xception_deeplabv3_aspp_bilinear_ade20k_ngpus1_batchSize8_trainCropSize512_LR_encoder1e-07_LR_decoder1e-07_epoch60+1110-13:57:38.log').read()
means = p.findall(s)
#s = file('../network/fcn8s_inceptionv2_p_pooling_ade20k_no_ave_no_stridedconv.log').read()
#means += p.findall(s)

acc1 = []
for i in means:
	acc1.append(i.strip().split(':')[-1])

s = file('./deeplabv3_pnorm_normsidebranch_no_conv1_poollrx1-10-30x10_initp70_scale100max110_pdiffx3_xception_deeplabv3_aspp_bilinear_ade20k_ngpus1_batchSize8_trainCropSize512_LR_encoder1e-07_LR_decoder1e-07_epoch60+1110-15:38:24.log').read()
means = p.findall(s)
#s = file('../network/vgg_ade20k_p_pooling_1.log').read()
#means += p.findall(s)

acc2 = []
for i in means:
	acc2.append(i.strip().split(':')[-1])

acc1 = (np.array(acc1).astype(float)*100)
acc2 = np.array(acc2).astype(float)*100
plt.plot(acc1)
plt.plot(acc2)
plt.show()

