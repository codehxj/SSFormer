import torch.nn.functional
from model import common
from model import attention
import torch
import torch.nn as nn

def make_model(args, parent=False):
    if args.dilation:
        # noinspection PyUnresolvedReferences
        from model import dilated
        return NLSN(args, dilated.dilated_conv)
    else:
        return NLSN(args)



class BasicConv(nn.Module):
    def __init__(self, in_feature, out_feature, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=False, bias=False):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_feature, out_feature, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_feature, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialAttentionLayer(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttentionLayer, self).__init__()
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, padding=kernel_size // 2, relu=False)

    def forward(self, x):
        scale = self.compress(x)
        scale = self.spatial(scale)
        scale = torch.sigmoid(scale)
        return x * scale


### Channel Attention from SE Network

class ChannelAttentionLayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttentionLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.avg_pool(x)
        scale = self.conv_fc(scale)
        return x * scale
class SDAB(nn.Module):
    # Sequential dual attention block similar to CBAM
    def __init__(self, n_feat, reduction, bias=True, bn=False):
        super(SDAB, self).__init__()
        conv = common.default_conv
        modules_head = []
        modules_head.append(conv(n_feat, n_feat, 3, bias=bias))
        modules_head.append(nn.ReLU())
        modules_head.append(conv(n_feat, n_feat, 3, bias=bias))
        self.head1 = nn.Sequential(*modules_head)

        self.spatial_attention = SpatialAttentionLayer()
        self.channel_attention = ChannelAttentionLayer(n_feat, reduction)

    def forward(self, x):
        out = self.head1(x)
        out = self.channel_attention(out)
        out = self.spatial_attention(out)
        out += x
        return out


class SDABRG(nn.Module):
    # Recursive group with sequential dual attention block
    def __init__(self, n_feat, reduction, n_dab):
        super(SDABRG, self).__init__()
        conv = common.default_conv
        modules_body = []
        for i in range(n_dab):
            modules_body.append(SDAB(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.conv_last = conv(n_feat, n_feat, 3)

    def forward(self, x):
        out = self.body(x)
        out = self.conv_last(out)
        out += x
        return out




# noinspection PyTypeChecker
class NLSN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(NLSN, self).__init__()

        self.args = args
        n_resblock = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3  
        scale = args.scale[0]
        act = nn.ReLU(True)

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)#(1,)
        m_head = [conv(args.n_colors, n_feats, kernel_size)]# n_feats=256 n_color = 3


        m_body = [attention.NonLocalSparseAttention(
            channels=n_feats, chunk_size=args.chunk_size, n_hashes=args.n_hashes, reduction=4, res_scale=args.res_scale)]         

        for i in range(n_resblock):
            m_body.append( common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ))
            if (i+1)%8==0:
                m_body.append(attention.NonLocalSparseAttention(
                    channels=n_feats, chunk_size=args.chunk_size, n_hashes=args.n_hashes, reduction=4, res_scale=args.res_scale))#n_feats=256
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module

        m_tail = [


            nn.Conv2d(
                n_feats, args.n_colors, kernel_size,stride=1,
                padding=(kernel_size//2)
            )
        ]
        m_tail1 = [

            nn.Conv2d(
                n_feats, 4 * n_feats, kernel_size,
                padding=(kernel_size // 2), stride=1, bias=True
            )


        ]
        m_tail2 = [
            nn.PixelShuffle(2)


        ]
        m_tail3 = [
            nn.PReLU(n_feats)
        ]
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 2)#1

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)#lshde huodong
        self.tail = nn.Sequential(*m_tail)
        self.tail1 = nn.Sequential(*m_tail1)
        self.tail2 = nn.Sequential(*m_tail2)
        self.tail3 = nn.Sequential(*m_tail3)

        self.attention=Blanced_attention.BlancedAttention(n_feats)
    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)#conv
        res = self.body(x)#nlsn + 8
        res = self.attention(res)
        b, h, _, w = x.size()
        z = x.permute(2,3 , 1, 0)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.m = SDABRG(w, 8, 8).to(device)
        t = self.m(z)
        t = t.permute(3, 2, 0, 1)
        y=res+t
        y = self.out_conv(y)
        x = x+y
        x = self.tail1(x)  
        x = self.tail2(x)  
        x = self.tail(x)
        x = self.add_mean(x)
        return x 

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

