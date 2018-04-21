import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F


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

    def __init__(self, in_channels, out_channels, pad, dilate=2, kernel_width, *args, **kwargs):
        pad = dilate if pad is None else pad
        super().__init__(
            in_channels, out_channels, ksize=[1, kernel_width], pad=[0, pad], dilate=[1, dilate], *args, **kwargs
        )
        self.crop = 2 * pad - dilate

    def __call__(self, x):
        ret = super().__call__(x)
        return ret[:, :, :, :-self.crop]  # B, C, 1, W


class DilTCN(chainer.chain):
    
    def __init__(self):
        super(DilTCN, self).__init__(
        conv1=CausalDilatedConvolution1D(),
        conv2= stacks=StackList(stacks_num, layers_num, hidden_dim, hidden_dim, kernel_width),
        conv2=L.Convolution2D(hidden_dim, out_hidden_dim, 1, initialW=INIT),
        conv3=L.Convolution2D(out_hidden_dim, out_channels, 1, initialW=INIT),
        )
        self.out_channels = out_channels

    def __call__(self, x, label):
        x, skip = self.stacks(self.conv1(x))
        x = self.conv2(F.relu(x+sum(skip)))
        x = self.conv3(F.relu(x))

        batch_size, _, _, width = x.shape
        x = F.reshape(x, [batch_size, self.out_channels, 1, 1, width])
        return x
