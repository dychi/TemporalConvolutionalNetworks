import numpy as np
import chainer.links as L
import chainer.functions as F
import chainer


class EDTCN(chainer.chain):
    
    def __init__(self):
        super(EDTCN, self).__init__(
        conv1=L.Convolution1D(in_channels=3, out_channels=64, ksize=3, stride=1, pad=1),
        conv2=L.Convolution1D(),

        )

    def __call__(self, x):
        bottleneck = [] 
        h = F.max_pooling_1d(self.conv1(x)),
        h = F.max_pooling_1d(self.conv2(x))
        bottleneck = h


class CausalDilatedConvolution1D(chainer.links.DilatedConvolution2D):
    def __init__(self, 

class DilTCN(chainer.chain):
    
    def __init__(self):
        super(DilTCN, self).__init__(
        conv1=
