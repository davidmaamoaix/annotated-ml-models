# YOLOv3: An Incremental Improvement
#
# https://arxiv.org/abs/1804.02767v1

import torch as t
import numpy as np


# a simple residual unit consisting of 2 dbl_units
class ResidualUnit(t.nn.Module):

    def __init__(self, in_channels):
        # in_channels specifies the input channel;
        # conv_1 has out channels of 0.5 * in_channels
        # conv_2 has out channels of in_channels

        super(ResidualUnit, self).__init__()

        # the conv_1 layer has a kernel of size 1, and therefore
        # has no padding
        self.conv_1 = dbl_unit(in_channels, in_channels // 2, 1, pad=0)
        self.conv_2 = dbl_unit(in_channels // 2, in_channels)

    def forward(self, x):
        conv_1_output = self.conv_1(x)
        conv_2_output = self.conv_2(conv_1_output)

        return conv_2_output + x


# a full residual block, consisting of one dbl_unit and
# numerous ResidualUnit; the out_channels of ResidualBlock
# is 2 * in_channels
class ResidualBlock(t.nn.Module):

    def __init__(self, in_channels, n_res_unit=1):
        super(ResidualBlock, self).__init__()

        # the first layer is a dbl_unit whose out_channels is
        # 2 * in_channels of the entire block; note that stride = 2
        self.conv_1 = dbl_unit(in_channels, in_channels * 2, stride=2)

        res_list = [ResidualUnit(in_channels * 2) for i in range(n_res_unit)]
        self.res_units = t.nn.Sequential(*res_list)

    def forward(self, x):
        conv_1_output = self.conv_1(x)

        return self.res_units(conv_1_output)


# Darknetconv2D_BN_Leaky is the primary building block
# of the yolo-v3 network; it consists of a convolution
# layer with batch normalization and a leaky relu
# activation function
def dbl_unit(in_channels=3, out_channels=64, kernel_size=3, stride=1, pad=1):
    dbl_block = t.nn.Sequential()

    # constructs the convolution layer
    conv = t.nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        pad,
        bias=False # set bias to False since batch normalization exists
    )
    dbl_block.add_module('darknet_conv2d', conv)

    # batch norm layer
    #
    # note that the batch norm layer comes before the leaky ReLU activation
    # due to leaky ReLU's nonlinearity. This results in a more stable
    # distribution (source: https://arxiv.org/abs/1502.03167)
    batch_norm = t.nn.BatchNorm2d(out_channels)
    dbl_block.add_module('batch_norm', batch_norm)

    # leaky ReLU is a variant of ReLU that uses a given slope when x < 0
    leaky_relu = t.nn.LeakyReLU(0.1)
    dbl_block.add_module('leaky_relu', leaky_relu)

    return dbl_block


# 5 dbl_units bunched together, with alternating kernel size and channels:
# 0.5 * in_channels for layer 1, 3, and 5, and in_channels for layer 2 and 4
#
# used after every concatenation
#
# special: int, the in_channels of the first dbl_unit (if different
# from the in_channels parameter)
def dbl_bundle(in_channels, special=None):
    bundle = t.nn.Sequential(
        dbl_unit(
            special if special else in_channels, in_channels // 2, 1, pad=0
        ),
        dbl_unit(in_channels // 2, in_channels),
        dbl_unit(in_channels, in_channels // 2, 1, pad=0),
        dbl_unit(in_channels // 2, in_channels),
        dbl_unit(in_channels, in_channels // 2, 1, pad=0)
    )

    return bundle


def upsample_unit(scale_factor=2):
    return t.nn.Upsample(scale_factor=scale_factor)


class YOLO_V3(t.nn.Module):
    """This is the baseline of YOLO V3"""

    def __init__(self, img_size=416):
        # img_size: int, the dimension of the input (default: 416)

        # YOLO comes in 3 sizes: 320, 416 and 608; this implementation
        # uses the medium one

        super(YOLO_V3, self).__init__()
        self.img_size = img_size

        # model begins with a convolution layer with 32 output channels
        self.initial_conv = dbl_unit(3, 32)

        # first 3 residual block; the separation from the rest is
        # for making the routing of output layers easier
        #
        # note that the output of each residual block is 2 times
        # its in_channels
        self.residual_1 = t.nn.Sequential(
            ResidualBlock(32, 1),
            ResidualBlock(64, 2),
            ResidualBlock(128, 8)
        )

        self.residual_2 = ResidualBlock(256, 8)
        self.residual_3 = ResidualBlock(512, 4)

        # layers responsible for the y1 output: 5 * conv (DBL) layers,
        # another conv (DBL) layer, and a linear conv layer
        self.y1_bundle = dbl_bundle(1024)
        self.y1_last = t.nn.Sequential(
            dbl_unit(512, 1024),
            t.nn.Conv2d(1024, 255, 1, padding=0)
        )

        # conv (DBL) and upsample
        self.y2_upsample = t.nn.Sequential(
            dbl_unit(512, 256, 1, pad=0),
            upsample_unit()
        )
        self.y2_bundle = dbl_bundle(512, special=768)
        self.y2_last = t.nn.Sequential(
            dbl_unit(256, 512),
            t.nn.Conv2d(512, 255, 1, padding=0)
        )

        self.y3_upsample = t.nn.Sequential(
            dbl_unit(256, 128, 1, pad=0),
            upsample_unit(),
        )
        self.y3_bundle = dbl_bundle(256, special=384)
        self.y3_last = t.nn.Sequential(
            dbl_unit(128, 256),
            t.nn.Conv2d(256, 255, 1, padding=0)
        )


    def forward(self, x):

        # first conv layer
        initial_conv_output = self.initial_conv(x)

        # Darknet-53's residual layers
        residual_1_output = self.residual_1(initial_conv_output)
        residual_2_output = self.residual_2(residual_1_output)
        residual_3_output = self.residual_3(residual_2_output)

        # y1 output
        y1_bundle_output = self.y1_bundle(residual_3_output)
        y1_last_output = self.y1_last(y1_bundle_output)

        # y2 output
        y2_upsample_output = self.y2_upsample(y1_bundle_output)
        y2_concat_output = t.cat((y2_upsample_output, residual_2_output), 1)
        y2_bundle_output = self.y2_bundle(y2_concat_output)
        y2_last_output = self.y2_last(y2_bundle_output)

        # y3 output
        y3_upsample_output = self.y3_upsample(y2_bundle_output)
        y3_concat_output = t.cat((y3_upsample_output, residual_1_output), 1)
        y3_bundle_output = self.y3_bundle(y3_concat_output)
        y3_last_output = self.y3_last(y3_bundle_output)

        return y3_last_output

a = YOLO_V3()
print(a(t.zeros(1, 3, 416, 416)).shape)