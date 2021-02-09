# YOLOv3: An Incremental Improvement
#
# https://arxiv.org/abs/1804.02767v1

import torch as t
import numpy as np


# the routing layer routes the output of a previous layer to it
#
# e.g. the output of a routing layer assigned to the 1st layer
# will output whatever the 1st layer outputs
#
# in the original implementation this also handles concatenation;
# however, this script separates the two for readability
class RouteLayer(t.nn.Module):

    def __init__(self, assigned_layer):
        super(RouteLayer, self).__init__()
        self.layer = assigned_layer

# concatenates the output of the target layers among the channels axis as
# a form of residual
class ConcatenateLayer(t.nn.Module):

    def __init__(self, assigned_layers):
        super(ConcatenateLayer, self).__init__()
        self.layers = assigned_layers


class YoloLayer(t.nn.Module):

    def __init__(self, anchors):
        super(YoloLayer, self).__init__()
        self.anchors = anchors


# a simple residual unit consisting of 2 dbl_units
class ResidualUnit(t.nn.Module):

    def __init__(self, in_channels):
        # in_channels specifies the input channel;
        # conv_1 has out channels of 0.5 * in_channels
        # conv_2 has out channels of in_channels

        super(ResidualUnit, self).__init__()

        # TODO: the padding doesn't seems right
        self.conv_1 = dbl_unit(in_channels, in_channels // 2, 1, pad=0)
        self.conv_2 = dbl_unit(in_channels // 2, in_channels)

    def forward(self, x):
        conv_1_output = self.conv_1(x)
        conv_2_output = self.conv_2(conv_1_output)
        print(x.shape, conv_2_output.shape)

        return conv_2_output + x


# a full residual block, consisting of one dbl_unit and
# numerous ResidualUnit
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


def upsample_unit(stride=2):
    return t.nn.Upsample(scale_factor=stride)


# the detection layer
def yolo_layer(anchors=[(10, 13), (16, 30), (33, 23)]):
    return YoloLayer(anchors)


class YOLO_V3(t.nn.Module):
    """This is the baseline of YOLO V3"""

    def __init__(self, img_size=416):
        # img_size: int, the dimension of the input (default: 416)

        # YOLO comes in 3 sizes: 320, 416 and 608; this implementation
        # uses the medium one

        super(YOLO_V3, self).__init__()
        self.img_size = img_size

