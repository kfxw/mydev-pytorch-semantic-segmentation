import torch
import torch.nn as nn
import torchvision
from lib.nn import SynchronizedBatchNorm2d
import pdb

class ModelBuilder():
    # custom weights initialization
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
	    #nn.init.normal_(m.weight.data, mean=0, std=0.01)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)
        #elif classname.find('Linear') != -1:
        #    m.weight.data.normal_(0.0, 0.0001)

    def build_decoder(self, arch='ppm_bilinear_deepsup',
                      fc_dim=512, num_class=150,
                      weights=''):
        if arch == 'c0_bilinear':
            net_decoder = C0Bilinear(
                num_class=num_class,
                fc_dim=fc_dim)
        elif arch == 'c1_bilinear_deepsup':
            net_decoder = C1BilinearDeepSup(
                num_class=num_class,
                fc_dim=fc_dim)
        elif arch == 'c1_bilinear':
            net_decoder = C1Bilinear(
                num_class=num_class,
                fc_dim=fc_dim)
        elif arch == 'ppm_bilinear':
            net_decoder = PPMBilinear(
                num_class=num_class,
                fc_dim=fc_dim)
        elif arch == 'ppm_bilinear_deepsup':
            net_decoder = PPMBilinearDeepsup(
                num_class=num_class,
                fc_dim=fc_dim)
        elif arch == 'upernet_lite':
            net_decoder = UPerNet(
                num_class=num_class,
                fc_dim=fc_dim,
                fpn_dim=256)
        elif arch == 'upernet':
            net_decoder = UPerNet(
                num_class=num_class,
                fc_dim=fc_dim,
                fpn_dim=512)
        elif arch == 'upernet_tmp':
            net_decoder = UPerNetTmp(
                num_class=num_class,
                fc_dim=fc_dim,
                fpn_dim=512)
        elif arch == 'deeplabv3_aspp_bilinear':
            net_decoder = DeeplabV3ASPPBilinear(
                num_class=num_class,
                fc_dim=fc_dim,
                dilation_rates=(6, 12, 18))
        elif arch == 'deeplabv3_aspp_bilinear_deepsup':
            net_decoder = DeeplabV3ASPPBilinearDeepSup(
                num_class=num_class,
                fc_dim=fc_dim,
                dilation_rates=(6, 12, 18))
        else:
            raise Exception('Architecture undefined!')

        net_decoder.apply(self.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_decoder')
            net_decoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_decoder


def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
            conv3x3(in_planes, out_planes, stride),
            SynchronizedBatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            )

# last conv and no 3x3, bilinear upsample
class C0Bilinear(nn.Module):
    def __init__(self, num_class=150, fc_dim=2048):
        super(C0Bilinear, self).__init__()

        # last conv
        self.conv_last = nn.Conv2d(fc_dim, num_class, 1, 1, 0)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]
        x = self.conv_last(conv5)
	x = nn.functional.upsample(x, size=segSize, mode='bilinear', align_corners=False)
        #x = nn.functional.softmax(x, dim=1)

        return x

# last conv, bilinear upsample
class C1BilinearDeepSup(nn.Module):
    def __init__(self, num_class=150, fc_dim=2048):
        super(C1BilinearDeepSup, self).__init__()

        self.cbr = conv3x3_bn_relu(fc_dim, fc_dim // 4, 1)
        self.cbr_deepsup = conv3x3_bn_relu(fc_dim // 2, fc_dim // 4, 1)

        # last conv
        self.conv_last = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)
        self.conv_last_deepsup = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        x = self.cbr(conv5)
        x = self.conv_last(x)

        # deep sup
        conv4 = conv_out[-2]
        _ = self.cbr_deepsup(conv4)
        _ = self.conv_last_deepsup(_)

	x = nn.functional.upsample(x, size=segSize, mode='bilinear', align_corners=False)
        x = nn.functional.softmax(x, dim=1)
	_ = nn.functional.upsample(_, size=segSize, mode='bilinear', align_corners=False)
        _ = nn.functional.softmax(_, dim=1)

        return (x, _)


# last conv, bilinear upsample
class C1Bilinear(nn.Module):
    def __init__(self, num_class=150, fc_dim=2048):
        super(C1Bilinear, self).__init__()

        self.cbr = conv3x3_bn_relu(fc_dim, fc_dim // 4, 1)

        # last conv
        self.conv_last = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]
        x = self.cbr(conv5)
        x = self.conv_last(x)

        x = nn.functional.upsample(x, size=segSize, mode='bilinear', align_corners=False)
        x = nn.functional.softmax(x, dim=1)

        return x


# pyramid pooling, bilinear upsample
class PPMBilinear(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096, pool_scales=(1, 2, 3, 6)):
        super(PPMBilinear, self).__init__()

        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                SynchronizedBatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm = nn.ModuleList(self.ppm)

        self.conv_last = nn.Sequential(
            nn.Conv2d(fc_dim+len(pool_scales)*512, 512,
                      kernel_size=3, padding=1, bias=False),
            SynchronizedBatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_class, kernel_size=1)
        )

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.upsample(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)

        x = self.conv_last(ppm_out)

        x = nn.functional.upsample(x, size=segSize, mode='bilinear', align_corners=False)
        x = nn.functional.softmax(x, dim=1)
        return x


# pyramid pooling, bilinear upsample
class PPMBilinearDeepsup(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096, pool_scales=(1, 2, 3, 6)):
        super(PPMBilinearDeepsup, self).__init__()

        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                SynchronizedBatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm = nn.ModuleList(self.ppm)
        self.cbr_deepsup = conv3x3_bn_relu(fc_dim // 2, fc_dim // 4, 1)

        self.conv_last = nn.Sequential(
            nn.Conv2d(fc_dim+len(pool_scales)*512, 512,
                      kernel_size=3, padding=1, bias=False),
            SynchronizedBatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_class, kernel_size=1)
        )
        self.conv_last_deepsup = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)
        self.dropout_deepsup = nn.Dropout2d(0.1)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.upsample(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)

        x = self.conv_last(ppm_out)

        # deep sup
        conv4 = conv_out[-2]
        _ = self.cbr_deepsup(conv4)
        _ = self.dropout_deepsup(_)
        _ = self.conv_last_deepsup(_)

	x = nn.functional.upsample(x, size=segSize, mode='bilinear', align_corners=False)
        x = nn.functional.softmax(x, dim=1)
	_ = nn.functional.upsample(_, size=segSize, mode='bilinear', align_corners=False)
        _ = nn.functional.softmax(_, dim=1)

        return (x, _)


# upernet
class UPerNet(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096, pool_scales=(1, 2, 3, 6),
                 fpn_inplanes=(256,512,1024,2048), fpn_dim=256):
        super(UPerNet, self).__init__()

        # PPM Module
        self.ppm_pooling = []
        self.ppm_conv = []

        for scale in pool_scales:
            self.ppm_pooling.append(nn.AdaptiveAvgPool2d(scale))
            self.ppm_conv.append(nn.Sequential(
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                SynchronizedBatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm_pooling = nn.ModuleList(self.ppm_pooling)
        self.ppm_conv = nn.ModuleList(self.ppm_conv)
        self.ppm_last_conv = conv3x3_bn_relu(fc_dim + len(pool_scales)*512, fpn_dim, 1)

        # FPN Module
        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]: # skip the top layer
            self.fpn_in.append(nn.Sequential(
                nn.Conv2d(fpn_inplane, fpn_dim, kernel_size=1, bias=False),
                SynchronizedBatchNorm2d(fpn_dim),
                nn.ReLU(inplace=True)
            ))
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out = []
        for i in range(len(fpn_inplanes) - 1): # skip the top layer
            self.fpn_out.append(nn.Sequential(
                conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            ))
        self.fpn_out = nn.ModuleList(self.fpn_out)

        self.conv_last = nn.Sequential(
            conv3x3_bn_relu(len(fpn_inplanes) * fpn_dim, fpn_dim, 1),
            nn.Conv2d(fpn_dim, num_class, kernel_size=1)
        )

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale, pool_conv in zip(self.ppm_pooling, self.ppm_conv):
            ppm_out.append(pool_conv(nn.functional.upsample(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False)))
        ppm_out = torch.cat(ppm_out, 1)
        f = self.ppm_last_conv(ppm_out)

        fpn_feature_list = [f]
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x) # lateral branch

            f = nn.functional.upsample(
                f, size=conv_x.size()[2:], mode='bilinear', align_corners=False) # top-down branch
            f = conv_x + f

            fpn_feature_list.append(self.fpn_out[i](f))

        fpn_feature_list.reverse() # [P2 - P5]
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]
        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(nn.functional.upsample(
                fpn_feature_list[i],
                output_size,
                mode='bilinear', align_corners=False))
        fusion_out = torch.cat(fusion_list, 1)
        x = self.conv_last(fusion_out)

        x = nn.functional.upsample(x, size=segSize, mode='bilinear', align_corners=False)
        x = nn.functional.softmax(x, dim=1)

        return x

# 2018-08-29, deeplab v3 aspp
class DeeplabV3ASPPBilinear(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096, dilation_rates=(6, 12, 18)):
        super(DeeplabV3ASPPBilinear, self).__init__()

	self.aspp0 = nn.Sequential(
                nn.Conv2d(fc_dim, 256, kernel_size=1, bias=False),
                SynchronizedBatchNorm2d(256),
		nn.ReLU(inplace=True),
		)
	self.aspp1 = nn.Sequential(
                nn.Conv2d(fc_dim, fc_dim, kernel_size=3, bias=False,
			dilation=dilation_rates[0], padding=dilation_rates[0], groups=fc_dim),  # depthwise
                SynchronizedBatchNorm2d(fc_dim),
		nn.ReLU(inplace=True),
                nn.Conv2d(fc_dim, 256, kernel_size=1, bias=False),
                SynchronizedBatchNorm2d(256),
		nn.ReLU(inplace=True),
		)
	self.aspp2 = nn.Sequential(
                nn.Conv2d(fc_dim, fc_dim, kernel_size=3, bias=False, 
			dilation=dilation_rates[1], padding=dilation_rates[1], groups=fc_dim),
                SynchronizedBatchNorm2d(fc_dim),
		nn.ReLU(inplace=True),
                nn.Conv2d(fc_dim, 256, kernel_size=1, bias=False),
                SynchronizedBatchNorm2d(256),
		nn.ReLU(inplace=True),
		)
	self.aspp3 = nn.Sequential(
                nn.Conv2d(fc_dim, fc_dim, kernel_size=3, bias=False, 
			dilation=dilation_rates[2], padding=dilation_rates[2], groups=fc_dim),
                SynchronizedBatchNorm2d(fc_dim),
		nn.ReLU(inplace=True),
                nn.Conv2d(fc_dim, 256, kernel_size=1, bias=False),
                SynchronizedBatchNorm2d(256),
		nn.ReLU(inplace=True),
		)
	self.gap = nn.Sequential(
		nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(fc_dim, 256, kernel_size=1, bias=False),
                SynchronizedBatchNorm2d(256),
		nn.ReLU(inplace=True),
		)

	self.cat_projection = nn.Sequential(
                nn.Conv2d(1280, 308, kernel_size=1, bias=False),
                SynchronizedBatchNorm2d(308),
		nn.ReLU(inplace=True),
		#nn.Dropout2d(p=0.5)
		)

	self.decoder_projection = nn.Sequential(
                nn.Conv2d(308, 308, kernel_size=3, bias=False, padding=1, groups=308),
                SynchronizedBatchNorm2d(308),
		nn.ReLU(inplace=True),
		nn.Conv2d(308, 256, kernel_size=1, bias=False),
                SynchronizedBatchNorm2d(256),
		nn.ReLU(inplace=True),
		nn.Conv2d(256, 256, kernel_size=3, bias=False, padding=1, groups=256),
                SynchronizedBatchNorm2d(256),
		nn.ReLU(inplace=True),
		nn.Conv2d(256, 256, kernel_size=1, bias=False),
                SynchronizedBatchNorm2d(256),
		nn.ReLU(inplace=True),
		nn.Conv2d(256, num_class, kernel_size=1, bias=True),
		)


    def forward(self, conv_out, segSize=None):
        x = conv_out[-1]
        input_size = x.size()

	# aspp
	aspp_ave = self.gap(x)
	aspp0 = self.aspp0(x)
	aspp1 = self.aspp1(x)
	aspp2 = self.aspp2(x)
	aspp3 = self.aspp3(x)
	# concat
	aspp_ave = nn.functional.upsample(aspp_ave, size=input_size[2:], mode='bilinear', align_corners=False)
	x = torch.cat([aspp_ave, aspp0, aspp1, aspp2, aspp3], dim=1)
	x = self.cat_projection(x)
	# decode
	x = self.decoder_projection(x)
	# resize and output
        x = nn.functional.upsample(x, size=segSize, mode='bilinear', align_corners=False)
        #x = nn.functional.softmax(x, dim=1)

        return x

# 2018-09-01, deeplab v3 aspp deep sup
class DeeplabV3ASPPBilinearDeepSup(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096, dilation_rates=(6, 12, 18)):
        super(DeeplabV3ASPPBilinear, self).__init__()

	self.aspp0 = nn.Sequential(
                nn.Conv2d(fc_dim, 256, kernel_size=1, bias=False),
                SynchronizedBatchNorm2d(256),
		nn.ReLU(inplace=True),
		)
	self.aspp1 = nn.Sequential(
                nn.Conv2d(fc_dim, fc_dim, kernel_size=3, bias=False,
			dilation=dilation_rates[0], padding=dilation_rates[0], groups=fc_dim),  # depthwise
                SynchronizedBatchNorm2d(fc_dim),
		nn.ReLU(inplace=True),
                nn.Conv2d(fc_dim, 256, kernel_size=1, bias=False),
                SynchronizedBatchNorm2d(256),
		nn.ReLU(inplace=True),
		)
	self.aspp2 = nn.Sequential(
                nn.Conv2d(fc_dim, fc_dim, kernel_size=3, bias=False, 
			dilation=dilation_rates[1], padding=dilation_rates[1], groups=fc_dim),
                SynchronizedBatchNorm2d(fc_dim),
		nn.ReLU(inplace=True),
                nn.Conv2d(fc_dim, 256, kernel_size=1, bias=False),
                SynchronizedBatchNorm2d(256),
		nn.ReLU(inplace=True),
		)
	self.aspp3 = nn.Sequential(
                nn.Conv2d(fc_dim, fc_dim, kernel_size=3, bias=False, 
			dilation=dilation_rates[2], padding=dilation_rates[2], groups=fc_dim),
                SynchronizedBatchNorm2d(fc_dim),
		nn.ReLU(inplace=True),
                nn.Conv2d(fc_dim, 256, kernel_size=1, bias=False),
                SynchronizedBatchNorm2d(256),
		nn.ReLU(inplace=True),
		)
	self.gap = nn.Sequential(
		nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(fc_dim, 256, kernel_size=1, bias=False),
                SynchronizedBatchNorm2d(256),
		nn.ReLU(inplace=True),
		)

	self.cat_projection = nn.Sequential(
                nn.Conv2d(1280, 256, kernel_size=1, bias=False),
                SynchronizedBatchNorm2d(308),
		nn.ReLU(inplace=True),
		nn.Dropout2d(p=0.5)
		)

	self.deepsup_projection = nn.Sequential(
                nn.Conv2d(1024, 52, kernel_size=1, bias=False),
                SynchronizedBatchNorm2d(52),
		nn.ReLU(inplace=True),
		)

	self.decoder_projection = nn.Sequential(
                nn.Conv2d(308, 308, kernel_size=3, bias=False, padding=1, groups=308),
                SynchronizedBatchNorm2d(308),
		nn.ReLU(inplace=True),
		nn.Conv2d(308, 256, kernel_size=1, bias=False),
                SynchronizedBatchNorm2d(256),
		nn.ReLU(inplace=True),
		nn.Conv2d(256, 256, kernel_size=3, bias=False, padding=1, groups=256),
                SynchronizedBatchNorm2d(256),
		nn.ReLU(inplace=True),
		nn.Conv2d(256, 256, kernel_size=1, bias=False),
                SynchronizedBatchNorm2d(256),
		nn.ReLU(inplace=True),
		nn.Conv2d(256, num_class, kernel_size=1, bias=True),
		)


    def forward(self, conv_out, segSize=None):
        x = conv_out[-1]
        input_size = x.size()
	deepsup = conv_out[-2]
	deepsup_size = deepsup.size()

	# aspp
	aspp_ave = self.gap(x)
	aspp0 = self.aspp0(x)
	aspp1 = self.aspp1(x)
	aspp2 = self.aspp2(x)
	aspp3 = self.aspp3(x)

	# concat
	aspp_ave = nn.functional.upsample(aspp_ave, size=input_size[2:], mode='bilinear', align_corners=False)
	x = torch.cat([aspp_ave, aspp0, aspp1, aspp2, aspp3], dim=1)
	# deep supervision projection and concat
	deepsup = self.deepsup_projection(deepsup)
	x = torch.cat([deepsup, x], dim=1)
	x = self.cat_projection(x)
	# decode
	x = self.decoder_projection(x)
	# resize and output
        x = nn.functional.upsample(x, size=segSize, mode='bilinear', align_corners=False)
        #x = nn.functional.softmax(x, dim=1)
        return x
