from blocks import *


class SimpleFFNN(Module):
    def __init__(self, _in=548, _out=7,
                 n_layers=4, n_inp=64, batch_normal=False, activation=None,
                 pre_funnel=False, post_funnel=False, softmax=False):
        super().__init__()
        self._out = _out
        blocks = []
        if pre_funnel:
            blocks.append(LinearFunnelDecoderBlock(_in=_in, _out=n_inp))
            # blocks.append(ReLU())
        else:
            n_inp = _in
        if not post_funnel:
            n_out = _out
        else:
            n_out = n_inp
            n_layers -= 1
        blocks.append(LinearConstantBlock(_in=n_inp, _out=n_out, n_layers=n_layers, n_inp=n_inp,
                                          batch_normal=batch_normal, activation=activation))
        if post_funnel:
            blocks.append(LinearFunnelDecoderBlock(_in=n_out, _out=_out))

        if softmax:
            self.fnn = Sequential(*blocks, Softmax(dim=-1))
        else:
            self.fnn = Sequential(*blocks)

    def forward(self, x):
        if self._out == 1:
            out = self.fnn(x)
            return out.clamp(0, 1)
        else:
            return self.fnn(x)


class SimpleCNN(Module):
    def __init__(self, _in=1, _out=7, n_layers=4, n_filters=64, d_1=False,
                 enc=False, dec=False, activation=ReLU(), softmax=False):
        super().__init__()
        blocks = []
        n_inp, n_out = _in, _out
        if enc:
            blocks.append(ConvFunnelEncoderBlock(_in=_in, _out=n_filters, d_1=d_1, ap=False))
            n_inp = n_filters
            n_layers -= 1
        if dec:
            n_out = n_filters
        blocks.append(ConvConstantEncoderBlock(_in=n_inp, _out=n_out, n_filters=n_filters, n_layers=n_layers, d_1=d_1))
        if dec:
            blocks.append(Flatten())
            blocks.append(LinearFunnelDecoderBlock(_in=n_out, _out=_out, activation=activation))

        if softmax:
            self.conv1d = Sequential(*blocks, Flatten(), Softmax(dim=1))
        else:
            self.conv1d = Sequential(*blocks, Flatten())

    def forward(self, x):
        return self.conv1d(x)


class SplitCNN(Module):
    def __init__(self, _splits, _in=1, _out=7, n_filters=512, d_1=True):
        super().__init__()
        self.splits = _splits
        self.enc1 = ConvFunnelEncoderBlock(_in=_in, _out=n_filters // 2, d_1=d_1, ap=True)
        self.enc2 = ConvFunnelEncoderBlock(_in=_in, _out=n_filters // 2, d_1=d_1, ap=True)
        self.f1, self.f2 = Flatten(), Flatten()
        self.dec = LinearFunnelDecoderBlock(_in=n_filters, _out=_out, activation=ReLU())

    def forward(self, x):
        x0 = x[:, :, :self.splits[0]]
        x1 = x[:, :, self.splits[0]:]

        x0 = self.enc1(x0)
        x1 = self.enc2(x1)

        x_cat = torch.cat((self.f1(x0), self.f2(x1)), 1)
        out = self.dec(x_cat)
        return out


class ConstantCNNEnc(Module):
    def __init__(self, _in=1, _out=7, n_layers=3, n_filters=32):
        super().__init__()
        self.cnn = ConvConstantEncoderBlock(_in=_in, _out=_out, n_layers=n_layers, n_filters=n_filters)

    def forward(self, x):
        return self.fnn(x)


class ConstantCNNAutoEnc(Module):
    def __init__(self, _in=1, _out=7, n_layers=3, n_filters=32):
        super().__init__()
        enc = ConvConstantEncoderBlock(_in=_in, _out=n_filters, n_layers=n_layers, n_filters=n_filters)
        dec = LinearFunnelDecoderBlock(_in=n_filters, _out=_out)
        self.net = Sequential(enc, dec)

    def forward(self, x):
        return self.net(x)


class FunnelCNNAutoEnc(Module):
    def __init__(self, _in=1, _out=7, n_filters=512, activation=None, leaky=False, maxpool=False, dropout=0,
                 softmax=False, d_1=True):
        super().__init__()
        enc = ConvFunnelEncoderBlock(_in=_in, _out=n_filters, ap=True, leaky=leaky, maxpool=maxpool, dropout=dropout, d_1=d_1)
        dec = LinearFunnelDecoderBlock(_in=n_filters, _out=_out, activation=activation)
        if softmax:
            self.net = Sequential(enc, Flatten(), dec, Softmax())
        else:
            self.net = Sequential(enc, Flatten(), dec)

    def forward(self, x):
        return self.net(x)


class VGG(Module):
    def __init__(self, _in=1, _out=7):
        super().__init__()
        self.vgg = Sequential(
            Conv1d(1, 64, kernel_size=3, padding=1),
            ReLU(inplace=True),
            Conv1d(64, 64, kernel_size=3, padding=1),
            ReLU(inplace=True),
            MaxPool1d(kernel_size=2, stride=2),
            Conv1d(64, 128, kernel_size=3, padding=1),
            ReLU(inplace=True),
            Conv1d(128, 128, kernel_size=3, padding=1),
            ReLU(inplace=True),
            MaxPool1d(kernel_size=2, stride=2),
            Conv1d(128, 256, kernel_size=3, padding=1),
            ReLU(inplace=True),
            Conv1d(256, 256, kernel_size=3, padding=1),
            ReLU(inplace=True),
            Conv1d(256, 256, kernel_size=3, padding=1),
            ReLU(inplace=True),
            MaxPool1d(kernel_size=2, stride=2),
            Conv1d(256, 512, kernel_size=3, padding=1),
            ReLU(inplace=True),
            Conv1d(512, 512, kernel_size=3, padding=1),
            ReLU(inplace=True),
            Conv1d(512, 512, kernel_size=3, padding=1),
            ReLU(inplace=True),
            MaxPool1d(kernel_size=2, stride=2),
            Conv1d(512, 512, kernel_size=3, padding=1),
            ReLU(inplace=True),
            Conv1d(512, 512, kernel_size=3, padding=1),
            ReLU(inplace=True),
            Conv1d(512, 512, kernel_size=3, padding=1),
            ReLU(inplace=True),
            MaxPool1d(kernel_size=2, stride=2)
        )
        self.ap = AdaptiveAvgPool1d(1)
        self.flat = Flatten()
        self.fc = Linear(512, _out)

    def forward(self, x):
        vgg = self.vgg(x)
        ap = self.ap(vgg)
        fc = self.fc(self.flat(ap))
        return fc

class VGG2(Module):
    def __init__(self, _in=1, _out=7):
        super().__init__()
        self.vgg = Sequential(
            Conv2d(1, 64, kernel_size=3, padding=1),
            ReLU(inplace=True),
            Conv2d(64, 64, kernel_size=3, padding=1),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(64, 128, kernel_size=3, padding=1),
            ReLU(inplace=True),
            Conv2d(128, 128, kernel_size=3, padding=1),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(128, 256, kernel_size=3, padding=1),
            ReLU(inplace=True),
            Conv2d(256, 256, kernel_size=3, padding=1),
            ReLU(inplace=True),
            Conv2d(256, 256, kernel_size=3, padding=1),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(256, 512, kernel_size=3, padding=1),
            ReLU(inplace=True),
            Conv2d(512, 512, kernel_size=3, padding=1),
            ReLU(inplace=True),
            Conv2d(512, 512, kernel_size=3, padding=1),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(512, 512, kernel_size=3, padding=1),
            ReLU(inplace=True),
            Conv2d(512, 512, kernel_size=3, padding=1),
            ReLU(inplace=True),
            Conv2d(512, 512, kernel_size=3, padding=1),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2)
        )
        self.ap = AdaptiveAvgPool2d(1)
        self.flat = Flatten()
        self.fc = Linear(512, _out)

    def forward(self, x):
        vgg = self.vgg(x)
        ap = self.ap(vgg)
        fc = self.fc(self.flat(ap))
        return fc


class ResNet(Module):
    def __init__(self, block, layers, num_classes=7):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = Sequential(
            Conv1d(1, 64, kernel_size=7, stride=2, padding=3),
            BatchNorm1d(64),
            ReLU())
        self.maxpool = MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = AdaptiveAvgPool1d(1)
        self.flat = Flatten()
        # self.fc = LinearFunnelDecoderBlock(512, num_classes)
        self.fc = Linear(512, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = Sequential(
                Conv1d(self.inplanes, planes, kernel_size=1, stride=stride),
                BatchNorm1d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = self.flat(x)
        x = self.fc(x)

        return x


class InceptionNet(Module):
    def __init__(self, Out_Classes):
        super().__init__()
        self.Conv_1 = InceptConvBlock(In_Channels=1, Out_Channels=64, Kernel_Size=7, Stride=2, Padding=3)
        self.MaxPool_1 = MaxPool1d(kernel_size=3, stride=2, padding=0, ceil_mode=True)
        self.Conv_2 = InceptConvBlock(In_Channels=64, Out_Channels=64, Kernel_Size=1, Stride=1, Padding=0)
        self.Conv_3 = InceptConvBlock(In_Channels=64, Out_Channels=192, Kernel_Size=3, Stride=1, Padding=1)
        self.MaxPool_2 = MaxPool1d(kernel_size=3, stride=2, padding=0, ceil_mode=True)
        self.Inception_3a = InceptionBlock(In_Channels=192, Num_Of_Filters_1x1=64, Num_Of_Filters_3x3=128,
                                           Num_Of_Filters_5x5=32, Num_Of_Filters_3x3_Reduce=96,
                                           Num_Of_Filters_5x5_Reduce=16, Pooling=32)

        self.Inception_3b = InceptionBlock(In_Channels=256, Num_Of_Filters_1x1=128, Num_Of_Filters_3x3=192
                                           , Num_Of_Filters_5x5=96, Num_Of_Filters_3x3_Reduce=128,
                                           Num_Of_Filters_5x5_Reduce=32, Pooling=64)

        self.MaxPool_3 = MaxPool1d(kernel_size=3, stride=2, padding=0, ceil_mode=True)
        self.Inception_4a = InceptionBlock(In_Channels=480, Num_Of_Filters_1x1=192, Num_Of_Filters_3x3=208
                                           , Num_Of_Filters_5x5=48, Num_Of_Filters_3x3_Reduce=96,
                                           Num_Of_Filters_5x5_Reduce=16, Pooling=64)

        self.Inception_4b = InceptionBlock(In_Channels=512, Num_Of_Filters_1x1=160, Num_Of_Filters_3x3=224
                                           , Num_Of_Filters_5x5=64, Num_Of_Filters_3x3_Reduce=112,
                                           Num_Of_Filters_5x5_Reduce=24, Pooling=64)

        self.Inception_4c = InceptionBlock(In_Channels=512, Num_Of_Filters_1x1=128, Num_Of_Filters_3x3=256
                                           , Num_Of_Filters_5x5=64, Num_Of_Filters_3x3_Reduce=128,
                                           Num_Of_Filters_5x5_Reduce=24, Pooling=64)

        self.Inception_4d = InceptionBlock(In_Channels=512, Num_Of_Filters_1x1=112, Num_Of_Filters_3x3=288
                                           , Num_Of_Filters_5x5=64, Num_Of_Filters_3x3_Reduce=144,
                                           Num_Of_Filters_5x5_Reduce=32, Pooling=64)

        self.Inception_4e = InceptionBlock(In_Channels=528, Num_Of_Filters_1x1=256, Num_Of_Filters_3x3=320
                                           , Num_Of_Filters_5x5=128, Num_Of_Filters_3x3_Reduce=160,
                                           Num_Of_Filters_5x5_Reduce=32, Pooling=128)

        self.MaxPool_4 = MaxPool1d(kernel_size=3, stride=2, padding=0, ceil_mode=True)
        self.Inception_5a = InceptionBlock(In_Channels=832, Num_Of_Filters_1x1=256, Num_Of_Filters_3x3=320
                                           , Num_Of_Filters_5x5=128, Num_Of_Filters_3x3_Reduce=160,
                                           Num_Of_Filters_5x5_Reduce=32, Pooling=128)

        self.Inception_5b = InceptionBlock(In_Channels=832, Num_Of_Filters_1x1=384, Num_Of_Filters_3x3=384
                                           , Num_Of_Filters_5x5=128, Num_Of_Filters_3x3_Reduce=192,
                                           Num_Of_Filters_5x5_Reduce=48, Pooling=128)

        self.AvgPool_1 = AdaptiveAvgPool1d(output_size=1)
        self.DropOut = Dropout(p=0.4)
        self.FC = Linear(in_features=1024, out_features=Out_Classes)

        self.Auxiliary_4a = InceptAuxiliaryClassifier(In_Channels=512, Num_Classes=Out_Classes)
        self.Auxiliary_4d = InceptAuxiliaryClassifier(In_Channels=528, Num_Classes=Out_Classes)

    def forward(self, Tensor_Path):
        # print(Tensor_Path.shape)
        Tensor_Path = self.Conv_1(Tensor_Path)
        # print(Tensor_Path.shape)
        Tensor_Path = self.MaxPool_1(Tensor_Path)
        Tensor_Path = self.Conv_2(Tensor_Path)
        Tensor_Path = self.Conv_3(Tensor_Path)
        Tensor_Path = self.MaxPool_2(Tensor_Path)
        Tensor_Path = self.Inception_3a(Tensor_Path)
        Tensor_Path = self.Inception_3b(Tensor_Path)
        Tensor_Path = self.MaxPool_3(Tensor_Path)
        Tensor_Path = self.Inception_4a(Tensor_Path)
        Auxiliary_1 = self.Auxiliary_4a(Tensor_Path)
        Tensor_Path = self.Inception_4b(Tensor_Path)
        Tensor_Path = self.Inception_4c(Tensor_Path)
        Tensor_Path = self.Inception_4d(Tensor_Path)
        Auxiliary_2 = self.Auxiliary_4d(Tensor_Path)
        Tensor_Path = self.Inception_4e(Tensor_Path)
        Tensor_Path = self.MaxPool_4(Tensor_Path)
        Tensor_Path = self.Inception_5a(Tensor_Path)
        Tensor_Path = self.Inception_5b(Tensor_Path)
        Tensor_Path = self.AvgPool_1(Tensor_Path)
        Tensor_Path = torch.flatten(Tensor_Path, 1)
        Tensor_Path = self.DropOut(Tensor_Path)
        Tensor_Path = self.FC(Tensor_Path)

        # return Tensor_Path, Auxiliary_1, Auxiliary_2
        return Tensor_Path
