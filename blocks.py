import torch
from torch.nn import *
import math


class ConvNormalizedBlock(Module):
    def __init__(self, _in, _out, k_size=3, pad=1, d_1=False, leaky=False, dropout=0, maxpool=False):
        super().__init__()
        # Convolution Block with Relu and Batch Norm.
        if d_1:
            conv = Conv1d(_in, _out, kernel_size=k_size, stride=2, padding=pad)
            if not leaky:
                relu = ReLU()
            else:
                relu = LeakyReLU(0.1)
            bn = BatchNorm1d(_out)
        else:
            conv = Conv2d(_in, _out, kernel_size=k_size, stride=(2, 2), padding=pad)
            if not leaky:
                relu = ReLU()
            else:
                relu = LeakyReLU(0.1)
            bn = BatchNorm2d(_out)
        # Use Kaiming Initialization
        init.kaiming_normal_(conv.weight, a=0.1)
        # Zero Out The Bias
        conv.bias.data.zero_()
        if dropout > 0:
            do = Dropout(dropout)
            if maxpool:
                mp = MaxPool1d(kernel_size=k_size, stride=2, padding=pad)
                self.conv_block = Sequential(conv, relu, mp, bn, do)
            else:
                self.conv_block = Sequential(conv, relu, bn, do)
        else:
            if maxpool:
                mp = MaxPool1d(kernel_size=k_size, stride=2, padding=pad)
                self.conv_block = Sequential(conv, relu, mp, bn)
            else:
                self.conv_block = Sequential(conv, relu, bn)

    def forward(self, x):
        return self.conv_block(x)


class ConvMaxPooledBlock(Module):
    def __init__(self, _in, _out, k_size=3, pad=1, d_1=True):
        super().__init__()
        # Convolution Block with Relu and MaxPooling.
        if d_1:
            conv = Conv1d(_in, _out, kernel_size=k_size, stride=2, padding=pad)
            relu = ReLU()
            mp = MaxPool1d(kernel_size=2)
        else:
            conv = Conv2d(_in, _out, kernel_size=k_size, stride=(2, 2), padding=pad)
            relu = ReLU()
            mp = MaxPool2d(kernel_size=2)
        # Use Kaiming Initialization
        init.kaiming_normal_(conv.weight, a=0.1)
        # Zero Out The Bias
        conv.bias.data.zero_()
        self.conv_block = Sequential(conv, relu, mp)

    def forward(self, x):
        return self.conv_block(x)


class ConvConstantEncoderBlock(Module):
    def __init__(self, _in=1, _out=7, n_layers=4, n_filters=64, d_1=False, ap=True):
        super().__init__()
        conv_in = ConvNormalizedBlock(_in, n_filters, k_size=5, pad=2, d_1=d_1)
        inter_layers = []
        for i in range(n_layers-1):
            inter_layers.append(ConvNormalizedBlock(n_filters, n_filters, d_1=d_1))
        conv_out = ConvNormalizedBlock(n_filters, _out, d_1=d_1)
        if ap:
            if d_1:
                self.enc_block = Sequential(conv_in, *inter_layers, conv_out, AdaptiveAvgPool1d(output_size=1))
            elif not d_1:
                self.enc_block = Sequential(conv_in, *inter_layers, conv_out, AdaptiveAvgPool2d(output_size=1))
        else:
            self.enc_block = Sequential(conv_in, *inter_layers, conv_out)

    def forward(self, x):
        return self.enc_block(x)


class ConvFunnelEncoderBlock(Module):
    def __init__(self, _in=1, _out=512, d_1=True, ap=True, leaky=False, maxpool=False, dropout=0):
        super().__init__()
        n_init = self.next_power_of_2(_in)
        n_last = self.prev_power_of_2(_out)
        n_layers = self.get_n_layers(n_init, n_last)
        n_init = 8 if n_init < 8 else n_init

        conv_in = ConvNormalizedBlock(_in, n_init, k_size=5, pad=2, d_1=d_1,
                                      dropout=dropout, leaky=leaky, maxpool=maxpool)
        inter_layers = []
        for i in range(n_layers):
            inter_layers.append(ConvNormalizedBlock(n_init, self.next_power_of_2(n_init)*2, d_1=d_1,
                                                    dropout=dropout, leaky=leaky, maxpool=maxpool))
            n_init = self.next_power_of_2(n_init) * 2
        conv_out = ConvNormalizedBlock(n_init, _out, d_1=d_1, dropout=dropout, leaky=leaky)
        if ap:
            # self.enc_block = Sequential(conv_in, *inter_layers, conv_out, AdaptiveAvgPool2d(1))
            if d_1:
                self.enc_block = Sequential(conv_in, *inter_layers, conv_out, AdaptiveAvgPool1d(1))
            else:
                self.enc_block = Sequential(conv_in, *inter_layers, conv_out, AdaptiveAvgPool2d(1))
        else:
            self.enc_block = Sequential(conv_in, *inter_layers, conv_out)


    @staticmethod
    def prev_power_of_2(x):
        return 1 if x == 0 else (2 ** (x - 1).bit_length()) // 2

    @staticmethod
    def next_power_of_2(x):
        return 1 if x == 0 else (2 ** (x - 1).bit_length())

    def get_n_layers(self, n_init, n_last):
        n_init = 8 if n_init <= 8 else n_init
        n = 0
        while n_init < n_last:
            n_init = self.next_power_of_2(n_init) * 2
            n += 1
        return n

    def forward(self, x):
        return self.enc_block(x)


class LinearFunnelDecoderBlock(Module):
    def __init__(self, _in=548, _out=7, batch_normal=False, activation=None):
        super().__init__()
        n_init = self.prev_power_of_2(_in)
        n_last = self.next_power_of_2(_out)
        n_layers = self.get_n_layers(n_init, n_last)
        fc_in = Linear(_in, n_init)
        inter_layers = []
        for i in range(n_layers):
            inter_layers.append(Linear(n_init, n_init//2))
            n_init //= 2
            if batch_normal:
                inter_layers.append(BatchNorm1d(n_inp))
            if activation is not None:
                inter_layers.append(activation)
        fc_out = Linear(n_init, _out)
        self.lin_dec = Sequential(fc_in, *inter_layers, fc_out)

    @staticmethod
    def prev_power_of_2(x):
        return 1 if x == 0 else (2 ** (x - 1).bit_length()) // 2

    @staticmethod
    def next_power_of_2(x):
        return 1 if x == 0 else (2 ** (x - 1).bit_length())

    @staticmethod
    def get_n_layers(n_init, n_last):
        n = 0
        while n_init > n_last:
            n_init //= 2
            n += 1
        return n

    def forward(self, x):
        return self.lin_dec(x)


class LinearConstantBlock(Module):
    def __init__(self, _in=1, _out=7, n_layers=4, n_inp=64, batch_normal=False, activation=None):
        super().__init__()
        lin_in = Linear(_in, n_inp)
        inter_layers = []
        for i in range(n_layers):
            inter_layers.append(Linear(n_inp, n_inp))
            if batch_normal:
                inter_layers.append(BatchNorm1d(n_inp))
            if activation is not None:
                inter_layers.append(activation)
        lin_out = Linear(n_inp, _out)
        self.lin_block = Sequential(lin_in, *inter_layers, lin_out)

    def forward(self, x):
        return self.lin_block(x)


class LSTMBlock(Module):
    def __init__(self, _in=548, n_hidden=256, n_layers=2, _drop_out=0.5):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden

        self.lstm = LSTM(_in, n_hidden, n_layers, dropout=_drop_out, batch_first=True)
        self.dropout = Dropout(_drop_out)

    def forward(self, x, hidden):
        l_out, l_hidden = self.lstm(x, hidden)  # (batch, seq_len, n_features)
        out = self.dropout(l_out)  # (batch, seq_len, n_hidden*direction)
        return out, l_hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data

        hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        return hidden


class ResidualBlock(Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = Sequential(
            Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            BatchNorm1d(out_channels),
            ReLU())
        self.conv2 = Sequential(
            Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            BatchNorm1d(out_channels))
        self.downsample = downsample
        self.relu = ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class InceptConvBlock(Module):
    def __init__(self, In_Channels, Out_Channels, Kernel_Size, Stride, Padding):
        super().__init__()
        self.Conv = Conv1d(in_channels=In_Channels, out_channels=Out_Channels, kernel_size=Kernel_Size,
                              stride=Stride, padding=Padding)
        self.Batch_Norm = BatchNorm1d(num_features=Out_Channels)
        self.Activ_Func = ReLU()

    def forward(self, Tensor_Path):
        Tensor_Path = self.Conv(Tensor_Path)
        Tensor_Path = self.Batch_Norm(Tensor_Path)
        Tensor_Path = self.Activ_Func(Tensor_Path)

        return Tensor_Path


class InceptionBlock(Module):
    def __init__(self, In_Channels, Num_Of_Filters_1x1, Num_Of_Filters_3x3, Num_Of_Filters_5x5,
                 Num_Of_Filters_3x3_Reduce, Num_Of_Filters_5x5_Reduce, Pooling):
        super(InceptionBlock, self).__init__()
        # The In_Channels are the depth of tensor coming from previous layer
        # First block contains only filters with kernel size 1x1
        self.Block_1 = Sequential(
            InceptConvBlock(In_Channels=In_Channels, Out_Channels=Num_Of_Filters_1x1, Kernel_Size=1, Stride=1,
                      Padding=0))

        # Second Block contains filters with kernel size 1x1 followed by 3x3
        self.Block_2 = Sequential(
            InceptConvBlock(In_Channels=In_Channels, Out_Channels=Num_Of_Filters_3x3_Reduce, Kernel_Size=1,
                      Stride=1, Padding=0),
            InceptConvBlock(In_Channels=Num_Of_Filters_3x3_Reduce, Out_Channels=Num_Of_Filters_3x3, Kernel_Size=3,
                      Stride=1, Padding=1)
        )

        # Third Block same as second block unless we'll replace the 3x3 filter with 5x5
        self.Block_3 = Sequential(
            InceptConvBlock(In_Channels=In_Channels, Out_Channels=Num_Of_Filters_5x5_Reduce, Kernel_Size=1,
                      Stride=1, Padding=0),
            InceptConvBlock(In_Channels=Num_Of_Filters_5x5_Reduce, Out_Channels=Num_Of_Filters_5x5, Kernel_Size=5,
                      Stride=1, Padding=2)
        )

        # Fourth Block contains maxpooling layer followed by 1x1 filter
        self.Block_4 = Sequential(
            MaxPool1d(kernel_size=3, stride=1, padding=1),
            InceptConvBlock(In_Channels=In_Channels, Out_Channels=Pooling, Kernel_Size=1, Stride=1, Padding=0)
        )

    def forward(self, Tensor_Path):
        First_Block_Out = self.Block_1(Tensor_Path)
        Second_Block_Out = self.Block_2(Tensor_Path)
        Third_Block_Out = self.Block_3(Tensor_Path)
        Fourth_Block_Out = self.Block_4(Tensor_Path)

        Concatenated_Outputs = torch.cat([First_Block_Out, Second_Block_Out, Third_Block_Out, Fourth_Block_Out],
                                         dim=1)  # dim=1 because we want to concatenate in the depth dimension
        return Concatenated_Outputs


class InceptAuxiliaryClassifier(Module):
    def __init__(self, In_Channels=1, Num_Classes=7):
        super().__init__()
        self.Adaptive_AvgPool = AdaptiveAvgPool1d(output_size=4)
        self.Conv = Conv1d(in_channels=In_Channels, out_channels=128, kernel_size=1, stride=1,
                              padding=0)
        self.Activ_Func = ReLU()
        # in_features=2048 because we should flatten the input tensor which has shape of (batch, 4,4,128) so after flaten the tensor will be (batch, 4*4*128)
        # out_features=1024 this number from paper
        self.FC_1 = Linear(in_features=512, out_features=256)
        self.DropOut = Dropout(p=0.7)
        self.FC_2 = Linear(in_features=256, out_features=Num_Classes)

    def forward(self, Tensor_Path):
        Tensor_Path = self.Adaptive_AvgPool(Tensor_Path)
        Tensor_Path = self.Conv(Tensor_Path)
        Tensor_Path = self.Activ_Func(Tensor_Path)
        Tensor_Path = torch.flatten(Tensor_Path, 1)
        Tensor_Path = self.FC_1(Tensor_Path)
        Tensor_Path = self.DropOut(Tensor_Path)
        Tensor_Path = self.FC_2(Tensor_Path)

        return Tensor_Path