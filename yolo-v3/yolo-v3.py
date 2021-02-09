# YOLOv3: An Incremental Improvement
#
# https://arxiv.org/abs/1804.02767v1

import torch as t
import numpy as np


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

    # note that the batch norm layer comes before the leaky ReLU activation
    # due to leaky ReLU's nonlinearity. This results in a more stable
    # distribution (source: https://arxiv.org/abs/1502.03167)
    batch_norm = t.nn.BatchNorm2d(out_channels)
    dbl_block.add_module('batch_norm', batch_norm)

    # leaky ReLU is a variant of ReLU that uses a given slope when x < 0
    leaky_relu = t.nn.LeakyReLU(0.1)
    dbl_block.add_module('leaky_relu', leaky_relu)


def upsample_unit(stride=2):
    return t.nn.Upsample(scale_factor=stride)


class YOLO_V3(t.nn.Module):
    """This is the baseline of YOLO V3"""

    def __init__(self, img_size=416):
        # img_size: int, the dimension of the input (default: 416)

        # YOLO comes in 3 sizes: 320, 416 and 608; this implementation
        # uses the medium one

        self.img_size = img_size