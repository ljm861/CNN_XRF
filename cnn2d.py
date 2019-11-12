import torch
import torch.nn as nn
from collections import namedtuple

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

def conv_block(in_dim, out_dim, act_fn, stride=1, padding=1, dilation=1):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=stride, padding=padding, dilation=dilation),
        nn.InstanceNorm2d(out_dim),
        act_fn,
    )
    return model

def conv_block_3(in_dim, out_dim, act_fn, stride=1, padding=1, dilation=1):
    model = nn.Sequential(
        conv_block(in_dim, out_dim, act_fn, stride, padding, dilation),
        conv_block(out_dim, out_dim, act_fn, stride, padding, dilation),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.InstanceNorm2d(out_dim),
    )
    return model

def conv_dil_block(in_dim, out_dim, act_fn, dil):
    model = nn.Sequential(
        nn.ReplicationPad2d(dil),
        nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=0, dilation=dil),
        nn.InstanceNorm2d(out_dim),
        act_fn,
    )
    return model

def conv_dil_block_3(in_dim, out_dim, act_fn, dil):
    model = nn.Sequential(
        conv_dil_block(in_dim, out_dim, act_fn, dil),
        conv_dil_block(out_dim, out_dim, act_fn, dil),
        nn.ReplicationPad2d(dil),
        nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=0, dilation=dil),
        nn.InstanceNorm2d(out_dim),
    )
    return model

class Conv_residual_conv(nn.Module):
    def __init__(self, in_dim, out_dim, act_fn, dil):
        super(Conv_residual_conv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        act_fn = act_fn
        dil = dil

        self.conv_1 = conv_dil_block(self.in_dim, self.out_dim, act_fn, dil)
        self.conv_2 = conv_dil_block_3(self.out_dim, self.out_dim, act_fn, dil)
        self.conv_3 = conv_dil_block(self.out_dim, self.out_dim, act_fn, dil)

    def forward(self, input):
        conv_1 = self.conv_1(input)
        conv_2 = self.conv_2(conv_1)
        res = conv_1 + conv_2
        conv_3 = self.conv_3(res)
        return conv_3

class CNN2D(nn.Module):

    def __init__(self, args):
        super(CNN2D, self).__init__()

        self.in_dim = args.input_dim
        self.num_feature = args.num_feature
        self.final_out_dim = args.output_dim
        self.out_clamp = args.out_clamp

        act_fn = nn.LeakyReLU(0.2, inplace=True)
        act_fn_2 = nn.ELU(inplace=True)

        # increase feature map
        self.increase_1 = Conv_residual_conv(self.in_dim, self.num_feature, dil=1, act_fn=act_fn)
        self.increase_2 = Conv_residual_conv(self.num_feature, self.num_feature * 2, dil=2, act_fn=act_fn)
        self.increase_3 = Conv_residual_conv(self.num_feature * 2, self.num_feature * 4, dil=3, act_fn=act_fn)
        self.increase_4 = Conv_residual_conv(self.num_feature * 4, self.num_feature * 8, dil=4, act_fn=act_fn)

        # bridge
        self.bridge = Conv_residual_conv(self.num_feature * 8, self.num_feature * 8, dil=5, act_fn=act_fn)

        # decrease feature map
        self.decrease_1 = Conv_residual_conv(self.num_feature * 16, self.num_feature * 4, dil=4, act_fn=act_fn_2)
        self.decrease_2 = Conv_residual_conv(self.num_feature * 8, self.num_feature * 2, dil=3, act_fn=act_fn_2)
        self.decrease_3 = Conv_residual_conv(self.num_feature * 4, self.num_feature * 1, dil=2, act_fn=act_fn_2)
        self.decrease_4 = Conv_residual_conv(self.num_feature * 2, self.num_feature * 1, dil=1, act_fn=act_fn_2)

        # output
        self.out = nn.Conv2d(self.num_feature, self.final_out_dim, kernel_size=3, stride=1, padding=1)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.apply(weights_init_kaiming)
            elif isinstance(m, nn.BatchNorm2d):
                m.apply(weights_init_kaiming)

        print("------CNN2D Init Done------")

    def forward(self, input):
        increase_1 = self.increase_1(input)
        increase_2 = self.increase_2(increase_1)
        increase_3 = self.increase_3(increase_2)
        increase_4 = self.increase_4(increase_3)
        bridge = self.bridge(increase_4)

        concat_1 = torch.cat([bridge, increase_4], dim=1)
        decrease_1 = self.decrease_1(concat_1)
        concat_2 = torch.cat([decrease_1, increase_3], dim=1)
        decrease_2 = self.decrease_2(concat_2)
        concat_3 = torch.cat([decrease_2, increase_2], dim=1)
        decrease_3 = self.decrease_3(concat_3)
        concat_4 = torch.cat([decrease_3, increase_1], dim=1)
        decrease_4 = self.decrease_4(concat_4)

        output = self.out(decrease_4)

        if self.out_clamp is not None:
            output = torch.clamp(output, min=self.out_clamp[0], max=self.out_clamp[1])

        return output

if __name__ == "__main__":
    input_ = torch.randn(8, 1, 20, 20)

    Arg = namedtuple('Arg', ['input_dim', 'num_feature', 'output_dim', 'out_clamp'])
    args = Arg(1, 16, 1, None)

    m = CNN2D(args)
    output = m(input_)
    print("output shape : ", output.shape)
