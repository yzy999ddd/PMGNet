"""
## Multi-Stage Progressive Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## https://arxiv.org/abs/2102.02808
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
from torch.nn import Conv2d, Linear, BatchNorm2d
from torch.nn.functional import relu
from torch.nn import Module, Parameter, init
import numbers
from einops import rearrange
from einops.layers.torch import Rearrange


##########################################################################
def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


def conv_down(in_chn, out_chn, bias=False):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, padding=1, bias=bias)
    return layer


# def apply_complex(fr, fi, input):
#    return (fr(input.real)-fi(input.imag)) + 1j*(fr(input.imag)+fi(input.real))

def apply_complex(fr, fi, input, dtype=torch.complex64):
    return (fr(input.real) - fi(input.imag)).type(dtype) \
        + 1j * (fr(input.imag) + fi(input.real)).type(dtype)


class ComplexConv2d(Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 dilation=1, groups=1, bias=True):
        super(ComplexConv2d, self).__init__()
        self.conv_r = Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_i = Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input):
        return apply_complex(self.conv_r, self.conv_i, input)


def complex_relu(input):
    return relu(input.real).type(torch.complex64) + 1j * relu(input.imag).type(torch.complex64)


class _ComplexBatchNorm(Module):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_ComplexBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features, 3))
            self.bias = Parameter(torch.Tensor(num_features, 2))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features, dtype=torch.complex64))
            self.register_buffer('running_covar', torch.zeros(num_features, 3))
            self.running_covar[:, 0] = 1.4142135623730951
            self.running_covar[:, 1] = 1.4142135623730951
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_covar', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_covar.zero_()
            self.running_covar[:, 0] = 1.4142135623730951
            self.running_covar[:, 1] = 1.4142135623730951
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.constant_(self.weight[:, :2], 1.4142135623730951)
            init.zeros_(self.weight[:, 2])
            init.zeros_(self.bias)


class ComplexBatchNorm2d(_ComplexBatchNorm):

    def forward(self, input):
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training or (not self.training and not self.track_running_stats):
            # calculate mean of real and imaginary part
            # mean does not support automatic differentiation for outputs with complex dtype.
            mean_r = input.real.mean([0, 2, 3]).type(torch.complex64)
            mean_i = input.imag.mean([0, 2, 3]).type(torch.complex64)
            mean = mean_r + 1j * mean_i
        else:
            mean = self.running_mean

        if self.training and self.track_running_stats:
            # update running mean
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean \
                                    + (1 - exponential_average_factor) * self.running_mean

        input = input - mean[None, :, None, None]

        if self.training or (not self.training and not self.track_running_stats):
            # Elements of the covariance matrix (biased for train)
            n = input.numel() / input.size(1)
            Crr = 1. / n * input.real.pow(2).sum(dim=[0, 2, 3]) + self.eps
            Cii = 1. / n * input.imag.pow(2).sum(dim=[0, 2, 3]) + self.eps
            Cri = (input.real.mul(input.imag)).mean(dim=[0, 2, 3])
        else:
            Crr = self.running_covar[:, 0] + self.eps
            Cii = self.running_covar[:, 1] + self.eps
            Cri = self.running_covar[:, 2]  # +self.eps

        if self.training and self.track_running_stats:
            with torch.no_grad():
                self.running_covar[:, 0] = exponential_average_factor * Crr * n / (n - 1) \
                                           + (1 - exponential_average_factor) * self.running_covar[:, 0]

                self.running_covar[:, 1] = exponential_average_factor * Cii * n / (n - 1) \
                                           + (1 - exponential_average_factor) * self.running_covar[:, 1]

                self.running_covar[:, 2] = exponential_average_factor * Cri * n / (n - 1) \
                                           + (1 - exponential_average_factor) * self.running_covar[:, 2]

        # calculate the inverse square root the covariance matrix
        det = Crr * Cii - Cri.pow(2)
        s = torch.sqrt(det)
        t = torch.sqrt(Cii + Crr + 2 * s)
        inverse_st = 1.0 / (s * t)
        Rrr = (Cii + s) * inverse_st
        Rii = (Crr + s) * inverse_st
        Rri = -Cri * inverse_st

        input = (Rrr[None, :, None, None] * input.real + Rri[None, :, None, None] * input.imag).type(torch.complex64) \
                + 1j * (Rii[None, :, None, None] * input.imag + Rri[None, :, None, None] * input.real).type(
            torch.complex64)

        if self.affine:
            input = (self.weight[None, :, 0, None, None] * input.real + self.weight[None, :, 2, None,
                                                                        None] * input.imag + \
                     self.bias[None, :, 0, None, None]).type(torch.complex64) \
                    + 1j * (self.weight[None, :, 2, None, None] * input.real + self.weight[None, :, 1, None,
                                                                               None] * input.imag + \
                            self.bias[None, :, 1, None, None]).type(torch.complex64)

        return input


##########################################################################
##---------- Prompt Gen Module -----------------------
class PromptAttBlock(nn.Module):
    def __init__(self, prompt_dim=128,prompt_size_H=96, prompt_size_W=96):
        super(PromptAttBlock, self).__init__()
        self.MaxPool2d = nn.MaxPool2d(kernel_size=3)
        self.prompt_param = nn.Parameter(torch.rand(1, prompt_dim, prompt_size_H, prompt_size_W))  # Pc
        self.conv3x3 = nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        emb = x.mean(dim=(-2, -1))
        prompt_weights = F.sigmoid(emb)
        prompt_ = self.prompt_param.unsqueeze(0).repeat(B,1,1,1,1).squeeze(1) *  prompt_weights.unsqueeze(-1).unsqueeze(-1)  
        prompt = F.interpolate(prompt_, (H, W), mode="bilinear")
        prompt = self.conv3x3(prompt)
        return prompt


##########################################################################
## Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # (B,C,1,1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),  # conv2d(80,20, 1,1)
            nn.ReLU(inplace=True), 
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),  # conv2d(20,80,1,1)
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)  # (1,80,1,1)
        y = self.conv_du(y)  #
        return x * y  # （1，80，64，64）  # x （1，80，64，64）

##########################################################################
## Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)  # （1，80，64，64）
        res = self.CA(res)
        res += x
        return res


##########################################################################
## Supervised Attention Module
class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size, bias):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 3, kernel_size, bias=bias)
        self.conv3 = conv(3, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1 * x2
        x1 = x1 + x
        return x1, img

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape  

        qkv = self.qkv_dwconv(self.qkv(x))  
        q, k, v = qkv.chunk(3, dim=1) 

        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)  
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)  
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature  
        attn = attn.softmax(dim=-1) 

        out = (attn @ v) 

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w) 

        out = self.project_out(out)  
        return out


## Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

## U-Net

class Encoder3(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff, heads=[2, 2, 4, 4], num_blocks = [1,2,2,4] ,
                 ffn_expansion_factor=2.66, LayerNorm_type='WithBias'):
        super(Encoder3, self).__init__()

        self.encoder_level1 = [  # 12个transformer块
            TransformerBlock(dim=n_feat, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])]
        self.encoder_level2 = [
            TransformerBlock(dim=n_feat + scale_unetfeats, num_heads=heads[1],
                             ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])]
        self.encoder_level3 = [
            TransformerBlock(dim=n_feat + scale_unetfeats * 2, num_heads=heads[2],
                             ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])]

        self.encoder_level1 = nn.Sequential(*self.encoder_level1)
        self.encoder_level2 = nn.Sequential(*self.encoder_level2)
        self.encoder_level3 = nn.Sequential(*self.encoder_level3)


        self.down12 = DownSample(n_feat, scale_unetfeats)  # 80,48
        self.down23 = DownSample(n_feat + scale_unetfeats, scale_unetfeats) # 128,48
        self.down34 = DownSample(n_feat + scale_unetfeats * 2, scale_unetfeats)

        # Cross Stage Feature Fusion (CSFF)
        if csff:
            self.csff_enc1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
            self.csff_enc2 = nn.Conv2d(n_feat + scale_unetfeats, n_feat + scale_unetfeats, kernel_size=1, bias=bias)
            self.csff_enc3 = nn.Conv2d(n_feat + (scale_unetfeats * 2), n_feat + (scale_unetfeats * 2), kernel_size=1,
                                       bias=bias)

            self.csff_dec1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
            self.csff_dec2 = nn.Conv2d(n_feat + scale_unetfeats, n_feat + scale_unetfeats, kernel_size=1, bias=bias)
            self.csff_dec3 = nn.Conv2d(n_feat + (scale_unetfeats * 2), n_feat + (scale_unetfeats * 2), kernel_size=1,
                                       bias=bias)

    def forward(self, x, encoder_outs=None, decoder_outs=None):

        enc1 = self.encoder_level1(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc1 = enc1 + self.csff_enc1(encoder_outs[0]) + self.csff_dec1(decoder_outs[0])

        x = self.down12(enc1)

        enc2 = self.encoder_level2(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc2 = enc2 + self.csff_enc2(encoder_outs[1]) + self.csff_dec2(decoder_outs[1])

        x = self.down23(enc2)

        enc3 = self.encoder_level3(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc3 = enc3 + self.csff_enc3(encoder_outs[2]) + self.csff_dec3(decoder_outs[2])


        return [enc1, enc2, enc3]  # (1,80,64,128) (1,128,32,64) (1,176,16,32) (1,224,8,16)


class Decoder3(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats, heads=[2, 2, 4, 4], num_blocks = [1,2,2,4] ,
                 ffn_expansion_factor=2.66, LayerNorm_type='WithBias'):
        super(Decoder3, self).__init__()

        self.decoder_level3 = [  # 12个transformer块
            TransformerBlock(dim=n_feat + scale_unetfeats * 2, num_heads=heads[2],
                             ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])]
        self.decoder_level2 = [
            TransformerBlock(dim=n_feat + scale_unetfeats, num_heads=heads[1],
                             ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])]
        self.decoder_level1 = [
            TransformerBlock(dim=n_feat, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])]

        self.decoder_level3 = nn.Sequential(*self.decoder_level3)
        self.decoder_level2 = nn.Sequential(*self.decoder_level2)
        self.decoder_level1 = nn.Sequential(*self.decoder_level1)

        self.skip_attn1 = nn.Conv2d(n_feat, n_feat, 3, 1, 1)
        self.skip_attn2 = nn.Conv2d(n_feat + scale_unetfeats,n_feat + scale_unetfeats, 3, 1, 1)
        self.skip_attn3 = nn.Conv2d(n_feat + scale_unetfeats * 2, n_feat + scale_unetfeats * 2, 3, 1, 1)

        self.up21 = SkipUpSample(n_feat, scale_unetfeats)
        self.up32 = SkipUpSample(n_feat + scale_unetfeats, scale_unetfeats)
        self.up43 = SkipUpSample(n_feat + scale_unetfeats * 2, scale_unetfeats)
    
        self.prompt4 = PromptAttBlock(prompt_dim=176, prompt_size_H=32,prompt_size_W=32)  
        self.prompt5 = PromptAttBlock(prompt_dim=128, prompt_size_H=64,prompt_size_W=64)
        self.prompt6 = PromptAttBlock(prompt_dim=80, prompt_size_H=128, prompt_size_W=128)


        self.noise_level4 = TransformerBlock((n_feat + scale_unetfeats * 2) * 2, num_heads=heads[2],
                                             ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                             LayerNorm_type=LayerNorm_type)
        self.noise_level5 = TransformerBlock((n_feat + scale_unetfeats ) * 2, num_heads=heads[2],
                                             ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                             LayerNorm_type=LayerNorm_type)
        self.noise_level6 = TransformerBlock(n_feat  * 2, num_heads=heads[2],
                                             ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                             LayerNorm_type=LayerNorm_type)


        self.reduce_degrad_level4 = nn.Conv2d((n_feat + scale_unetfeats * 2 ) * 2, n_feat + scale_unetfeats * 2,
                                              kernel_size=1, bias=bias)  ## 352 176

        self.reduce_degrad_level5 = nn.Conv2d((n_feat + scale_unetfeats ) * 2, n_feat + scale_unetfeats ,
                                              kernel_size=1, bias=bias)  ## 没改动 # 448,176

        self.reduce_degrad_level6 = nn.Conv2d(n_feat  * 2, n_feat ,
                                              kernel_size=1, bias=bias)  ## 352 176

    def forward(self, outs):
        enc1, enc2, enc3 = outs



        dec3 = self.decoder_level3(enc3)  # 1,176,32,32
        prompt4 = self.prompt4(dec3)
        prompt4 = torch.cat([prompt4, dec3], 1)
        prompt4 = self.noise_level4(prompt4)
        dec3 = self.reduce_degrad_level4(prompt4)

        x = self.up32(dec3, self.skip_attn2(enc2))
        dec2 = self.decoder_level2(x)  # 1,128,64,64
        prompt5 = self.prompt5(dec2)
        prompt5 = torch.cat([prompt5, dec2], 1)
        prompt5 = self.noise_level5(prompt5)
        dec2 = self.reduce_degrad_level5(prompt5)

        x = self.up21(dec2,self.skip_attn1(enc1))
        dec1 = self.decoder_level1(x)  # 1,80,128,128
        prompt6 = self.prompt6(dec1)  # 1,128,64,64
        prompt6 = torch.cat([prompt6, dec1], 1)  # 1,256,64,64
        prompt6 = self.noise_level6(prompt6)  # 1,256,64,64
        dec1 = self.reduce_degrad_level6(prompt6)  # 1×1  # 1,128,64,64

        return [dec1, dec2, dec3]


class Encoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff, heads=[2, 2, 4, 4], num_blocks = [1,2,2,4] ,
                 ffn_expansion_factor=2.66, LayerNorm_type='WithBias'):
        super(Encoder, self).__init__()

        self.encoder_level1 = [  # 12个transformer块
            TransformerBlock(dim=n_feat, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])]
        self.encoder_level2 = [
            TransformerBlock(dim=n_feat + scale_unetfeats, num_heads=heads[1],
                             ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])]
        self.encoder_level3 = [
            TransformerBlock(dim=n_feat + scale_unetfeats * 2, num_heads=heads[2],
                             ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])]

        # # 顺序连接模型
        self.encoder_level1 = nn.Sequential(*self.encoder_level1)
        self.encoder_level2 = nn.Sequential(*self.encoder_level2)
        self.encoder_level3 = nn.Sequential(*self.encoder_level3)


        self.down12 = DownSample(n_feat, scale_unetfeats)  # 80,48
        self.down23 = DownSample(n_feat + scale_unetfeats, scale_unetfeats) # 128,48
        self.down34 = DownSample(n_feat + scale_unetfeats * 2, scale_unetfeats)

        # Cross Stage Feature Fusion (CSFF)
        if csff:
            self.csff_enc1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
            self.csff_enc2 = nn.Conv2d(n_feat + scale_unetfeats, n_feat + scale_unetfeats, kernel_size=1, bias=bias)
            self.csff_enc3 = nn.Conv2d(n_feat + (scale_unetfeats * 2), n_feat + (scale_unetfeats * 2), kernel_size=1,
                                       bias=bias)

            self.csff_dec1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
            self.csff_dec2 = nn.Conv2d(n_feat + scale_unetfeats, n_feat + scale_unetfeats, kernel_size=1, bias=bias)
            self.csff_dec3 = nn.Conv2d(n_feat + (scale_unetfeats * 2), n_feat + (scale_unetfeats * 2), kernel_size=1,
                                       bias=bias)

    def forward(self, x, encoder_outs=None, decoder_outs=None):

        enc1 = self.encoder_level1(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc1 = enc1 + self.csff_enc1(encoder_outs[0]) + self.csff_dec1(decoder_outs[0])

        x = self.down12(enc1)

        enc2 = self.encoder_level2(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc2 = enc2 + self.csff_enc2(encoder_outs[1]) + self.csff_dec2(decoder_outs[1])

        x = self.down23(enc2)

        enc3 = self.encoder_level3(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc3 = enc3 + self.csff_enc3(encoder_outs[2]) + self.csff_dec3(decoder_outs[2])

#         x = self.down34(enc3)

#         enc4 = self.encoder_level4(x)
#         if encoder_outs is not None:
#             enc4 = enc4 + self.csff_enc4(encoder_outs[3])

        return [enc1, enc2, enc3]  # (1,80,64,128) (1,128,32,64) (1,176,16,32) (1,224,8,16)


class Decoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats, heads=[2, 2, 4, 4], num_blocks = [1,2,2,4] ,
                 ffn_expansion_factor=2.66, LayerNorm_type='WithBias'):
        super(Decoder, self).__init__()

        self.decoder_level3 = [  # 12个transformer块
            TransformerBlock(dim=n_feat + scale_unetfeats * 2, num_heads=heads[2],
                             ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])]
        self.decoder_level2 = [
            TransformerBlock(dim=n_feat + scale_unetfeats, num_heads=heads[1],
                             ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])]
        self.decoder_level1 = [
            TransformerBlock(dim=n_feat, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])]

        self.decoder_level3 = nn.Sequential(*self.decoder_level3)
        self.decoder_level2 = nn.Sequential(*self.decoder_level2)
        self.decoder_level1 = nn.Sequential(*self.decoder_level1)

        self.skip_attn1 = nn.Conv2d(n_feat, n_feat, 3, 1, 1)
        self.skip_attn2 = nn.Conv2d(n_feat + scale_unetfeats,n_feat + scale_unetfeats, 3, 1, 1)
        self.skip_attn3 = nn.Conv2d(n_feat + scale_unetfeats * 2, n_feat + scale_unetfeats * 2, 3, 1, 1)

        self.up21 = SkipUpSample(n_feat, scale_unetfeats)
        self.up32 = SkipUpSample(n_feat + scale_unetfeats, scale_unetfeats)
        self.up43 = SkipUpSample(n_feat + scale_unetfeats * 2, scale_unetfeats)

        self.prompt4 = PromptAttBlock(prompt_dim=176, prompt_size_H=32,prompt_size_W=32)  # 224 8 确认 lin_dim怎么选取？
        self.prompt5 = PromptAttBlock(prompt_dim=128, prompt_size_H=64,prompt_size_W=64)
        self.prompt6 = PromptAttBlock(prompt_dim=80, prompt_size_H=128, prompt_size_W=128)

        self.noise_level4 = TransformerBlock((n_feat + scale_unetfeats * 2) * 2, num_heads=heads[2],
                                             ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                             LayerNorm_type=LayerNorm_type)
        self.noise_level5 = TransformerBlock((n_feat + scale_unetfeats ) * 2, num_heads=heads[2],
                                             ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                             LayerNorm_type=LayerNorm_type)
        self.noise_level6 = TransformerBlock(n_feat  * 2, num_heads=heads[2],
                                             ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                             LayerNorm_type=LayerNorm_type)

        self.reduce_degrad_level4 = nn.Conv2d((n_feat + scale_unetfeats * 2 ) * 2, n_feat + scale_unetfeats * 2,
                                              kernel_size=1, bias=bias)  ## 352 176

        self.reduce_degrad_level5 = nn.Conv2d((n_feat + scale_unetfeats ) * 2, n_feat + scale_unetfeats ,
                                              kernel_size=1, bias=bias)  ## 没改动 # 448,176

        self.reduce_degrad_level6 = nn.Conv2d(n_feat  * 2, n_feat ,
                                              kernel_size=1, bias=bias)  ## 352 176

    def forward(self, outs):
        enc1, enc2, enc3 = outs

        dec3 = self.decoder_level3(enc3)  # 1,176,32,32
        prompt4 = self.prompt4(dec3)
        prompt4 = torch.cat([prompt4, dec3], 1)
        prompt4 = self.noise_level4(prompt4)
        dec3 = self.reduce_degrad_level4(prompt4)

        x = self.up32(dec3, self.skip_attn2(enc2))
        dec2 = self.decoder_level2(x)  # 1,128,64,64
        prompt5 = self.prompt5(dec2)
        prompt5 = torch.cat([prompt5, dec2], 1)
        prompt5 = self.noise_level5(prompt5)
        dec2 = self.reduce_degrad_level5(prompt5)

        x = self.up21(dec2,self.skip_attn1(enc1))
        dec1 = self.decoder_level1(x)  # 1,80,128,128
        prompt6 = self.prompt6(dec1)  # 1,128,64,64
        prompt6 = torch.cat([prompt6, dec1], 1)  # 1,256,64,64
        prompt6 = self.noise_level6(prompt6)  # 1,256,64,64
        dec1 = self.reduce_degrad_level6(prompt6)  # 1×1  # 1,128,64,64

        return [dec1, dec2, dec3]


class HinEncoder(nn.Module):
    def __init__(self, n_feat, kernel_size, bias, scale_unetfeats, relu_slope):
        super(HinEncoder, self).__init__()
        

        self.encoder_level1 = [UNetConvBlock(n_feat, n_feat, relu_slope, use_HIN=True)]  # (80,80)
        self.encoder_level2 = [UNetConvBlock(n_feat, n_feat + scale_unetfeats, relu_slope, use_HIN=True)]  # (80,128)
        self.encoder_level3 = [UNetConvBlock(n_feat + scale_unetfeats, n_feat + scale_unetfeats * 2, relu_slope,
                                             use_HIN=True)]  # (128,176)
        self.encoder_level4 = [UNetConvBlock(n_feat + scale_unetfeats * 2, n_feat + scale_unetfeats * 3, relu_slope,
                                             use_HIN=True)]  # (176,224)

       
        self.encoder_level1 = nn.Sequential(*self.encoder_level1)
        self.encoder_level2 = nn.Sequential(*self.encoder_level2)
        self.encoder_level3 = nn.Sequential(*self.encoder_level3)
        self.encoder_level4 = nn.Sequential(*self.encoder_level4)

        self.down12 = conv_down(n_feat, n_feat, bias=False)  # (80,80)
        self.down23 = conv_down(n_feat + scale_unetfeats, n_feat + scale_unetfeats, bias=False)  # (128,128)
        self.down34 = conv_down(n_feat + scale_unetfeats * 2, n_feat + scale_unetfeats * 2, bias=False)  # (176,176)


   
    def forward(self, x, encoder_outs=None, decoder_outs=None):
        enc1 = self.encoder_level1(x)  # 16,80,64,64
        x = self.down12(enc1)  # 16,80,32,32

        enc2 = self.encoder_level2(x)  # 16,128,32,32
        x = self.down23(enc2)  # 16,128,16,16

        enc3 = self.encoder_level3(x)  # 16,176,16,16
        x = self.down34(enc3)  # 16,176,8,8

        enc4 = self.encoder_level4(x)  # 16,224,8,8

        return [enc1, enc2, enc3, enc4]



class ResDecoder(nn.Module):
    def __init__(self, n_feat, kernel_size, relu_slope, bias, scale_unetfeats, heads=[1, 2, 4, 4],num_refinement_blocks = 4,
                 ffn_expansion_factor=2.66, LayerNorm_type='WithBias', ):
        super(ResDecoder, self).__init__()

        self.decoder_level1 = [UNetConvBlock(n_feat + scale_unetfeats, n_feat, relu_slope, use_HIN=False)]
        self.decoder_level2 = [
            UNetConvBlock(n_feat + scale_unetfeats * 2, n_feat + scale_unetfeats, relu_slope, use_HIN=False)]
        self.decoder_level3 = [
            UNetConvBlock(n_feat + (scale_unetfeats * 3), n_feat + (scale_unetfeats * 2), relu_slope, use_HIN=False)]

        self.decoder_level1 = nn.Sequential(*self.decoder_level1)
        self.decoder_level2 = nn.Sequential(*self.decoder_level2)
        self.decoder_level3 = nn.Sequential(*self.decoder_level3)

        self.prompt1 = PromptAttBlock(prompt_dim=176, prompt_size_H=16, prompt_size_W=32)
        self.prompt2 = PromptAttBlock(prompt_dim=128, prompt_size_H=32, prompt_size_W=64)
        self.prompt3 = PromptAttBlock(prompt_dim=80, prompt_size_H=64, prompt_size_W=128)  


        self.noise_level1 = TransformerBlock((n_feat + scale_unetfeats * 2) * 2, num_heads=heads[2],
                                             ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                             LayerNorm_type=LayerNorm_type)
        self.noise_level2 = TransformerBlock((n_feat + scale_unetfeats) * 2, num_heads=heads[2],
                                             ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                             LayerNorm_type=LayerNorm_type)
        self.noise_level3 = TransformerBlock(n_feat * 2, num_heads=heads[2],
                                             ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                             LayerNorm_type=LayerNorm_type)


        self.reduce_degrad_level1 = nn.Conv2d((n_feat + scale_unetfeats * 2) * 2, n_feat + scale_unetfeats * 2,
                                              kernel_size=1, bias=bias)   # 448,176
        self.reduce_degrad_level2 = nn.Conv2d((n_feat + scale_unetfeats ) * 2, n_feat + scale_unetfeats ,
                                              kernel_size=1, bias=bias)  ## 352 176
        self.reduce_degrad_level3 = nn.Conv2d(n_feat * 2, n_feat, kernel_size=1,
                                              bias=bias)  ## 256,128

        self.skip_attn1 = nn.Conv2d(n_feat, n_feat + scale_unetfeats, 3, 1, 1)
        self.skip_attn2 = nn.Conv2d(n_feat + scale_unetfeats, n_feat + scale_unetfeats * 2, 3, 1, 1)
        self.skip_attn3 = nn.Conv2d(n_feat + scale_unetfeats * 2, n_feat + scale_unetfeats * 3, 3, 1, 1)

        self.up43 = nn.ConvTranspose2d(n_feat + scale_unetfeats * 3, n_feat + scale_unetfeats * 3, kernel_size=2,
                                       stride=2, bias=True)
        self.up32 = nn.ConvTranspose2d(n_feat + scale_unetfeats * 2, n_feat + scale_unetfeats * 2, kernel_size=2,
                                       stride=2, bias=True)
        self.up21 = nn.ConvTranspose2d(n_feat + scale_unetfeats, n_feat + scale_unetfeats, kernel_size=2, stride=2,
                                       bias=True)

    def forward(self, outs):
        enc1, enc2, enc3, enc4 = outs

        dec3 = self.skip_attn3(enc3) + self.up43(enc4)
        dec3 = self.decoder_level3(dec3)  # 1,176,16,32
        prompt1= self.prompt1(dec3)  # 1,176,16,32
        prompt1 = torch.cat([prompt1, dec3], 1)  # 1,352,16,32
        prompt1 = self.noise_level1(prompt1)  # 1,352,16,32
        dec3 = self.reduce_degrad_level1(prompt1)  # 1×1 # 1,176,16,32

        dec2 = self.skip_attn2(enc2) + self.up32(dec3)
        dec2 = self.decoder_level2(dec2)  # 1,128,32,64
        prompt2= self.prompt2(dec2)  # 1,128,32,64
        prompt2 = torch.cat([prompt2, dec2], 1)  # 1,256,32,64
        prompt2 = self.noise_level2(prompt2)  # 1,256,32,64
        dec2 = self.reduce_degrad_level2(prompt2)  # 1×1 1,128,32,64

        dec1 = self.skip_attn1(enc1) + self.up21(dec2)
        dec1 = self.decoder_level1(dec1)  # 1,80,64,128
        prompt3= self.prompt3(dec1)  # 1,128,64,64
        prompt3 = torch.cat([prompt3, dec1], 1)  # 1,256,32,64
        prompt3 = self.noise_level3(prompt3)  # 1,256,32,64
        dec1 = self.reduce_degrad_level3(prompt3)  # 1×1 1,128,32,64

        return [dec1, dec2, dec3]


##########################################################################
##---------- Resizing Modules ----------
class DownSample(nn.Module):
    def __init__(self, in_channels, s_factor):  # 80，48  # 128，48
        super(DownSample, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, in_channels + s_factor, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels + s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x


class SkipUpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(SkipUpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels + s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x, y):
        x = self.up(x)
        x = x + y
        return x


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope, use_HIN=False):
        super(UNetConvBlock, self).__init__()
        # self.downsample = downsample
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN

    def forward(self, x, enc=None, dec=None):
        out = self.conv_1(x)

        if self.use_HIN:
            out_1, out_2 = torch.chunk(out, 2, dim=1)
            out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))

        out += self.identity(x)
        return out

class ComplexNet(nn.Module):
    def __init__(self, in_channels):
        super(ComplexNet, self).__init__()
        kernel_size = 3
        padding = 1
        features = 80
        groups = 1
        layers = []
        kernel_size1 = 1
        '''
        #self.gamma = nn.Parameter(torch.zeros(1))
        '''
        self.conv1_1 = ComplexConv2d(in_channels=in_channels, out_channels=features, kernel_size=kernel_size,
                                     padding=padding, groups=groups, bias=False)
        self.conv1_2 = ComplexConv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=2,
                                     groups=groups, bias=False, dilation=2)
        self.conv1_3 = ComplexConv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1,
                                     groups=groups, bias=False)
        self.conv1_4 = ComplexConv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1,
                                     groups=groups, bias=False)
        self.conv1_5 = ComplexConv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=2,
                                     groups=groups, bias=False, dilation=2)
        self.conv1_6 = ComplexConv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1,
                                     groups=groups, bias=False)
        self.conv1_7 = ComplexConv2d(in_channels=features, out_channels=features, kernel_size=kernel_size,
                                     padding=padding, groups=groups, bias=False)
        self.conv1_8 = ComplexConv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1,
                                     groups=groups, bias=False)
        self.conv1_9 = ComplexConv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=2,
                                     groups=groups, bias=False, dilation=2)
        self.conv1_10 = ComplexConv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1,
                                      groups=groups, bias=False)
        self.conv1_11 = ComplexConv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1,
                                      groups=groups, bias=False)
        self.conv1_12 = ComplexConv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=2,
                                      groups=groups, bias=False, dilation=2)
        self.bn = ComplexBatchNorm2d(features)

    def forward(self, x):
        res = complex_relu(self.bn(self.conv1_1(x)))
        res = complex_relu(self.bn(self.conv1_2(res)))
        res = complex_relu(self.bn(self.conv1_3(res)))
        res = complex_relu(self.bn(self.conv1_4(res)))
        res = complex_relu(self.bn(self.conv1_5(res)))
        res = complex_relu(self.bn(self.conv1_6(res)))
        res = complex_relu(self.bn(self.conv1_7(res)))
        res = complex_relu(self.bn(self.conv1_8(res)))
        res = complex_relu(self.bn(self.conv1_9(res)))
        res = complex_relu(self.bn(self.conv1_10(res)))
        res = complex_relu(self.bn(self.conv1_11(res)))
        res = complex_relu(self.bn(self.conv1_12(res)))
        return res.abs()


class PMGNet(nn.Module):  #####################  80，48
    def __init__(self, in_c=3, out_c=3, n_feat=80, scale_unetfeats=48, scale_orsnetfeats=32, num_cab=8, kernel_size=3,
                 relu_slope=0.2, reduction=4, bias=False):
        super(PMGNet, self).__init__()

        act = nn.PReLU()
        self.shallow_feat1 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias),
                                           CAB(n_feat, kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat2 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias),
                                           CAB(n_feat, kernel_size, reduction, bias=bias, act=act))
        self.ComplexNet = ComplexNet(in_c)

        self.stage1_encoder = HinEncoder(n_feat, kernel_size, bias, scale_unetfeats, relu_slope)  # (80，3，48，0.2)
        self.stage1_decoder = ResDecoder(n_feat, kernel_size, relu_slope, bias, scale_unetfeats)

        self.stage2_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=True)
        self.stage2_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)

        self.stage3_encoder = Encoder3(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=True)
        self.stage3_decoder = Decoder3(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)

        # self.stage3_orsnet = ORSNet(n_feat, scale_orsnetfeats, kernel_size, reduction, act, bias, scale_unetfeats,
        #                             num_cab)

        self.sam12 = SAM(n_feat, kernel_size=1, bias=bias)
        self.sam23 = SAM(n_feat, kernel_size=1, bias=bias)

        self.concat12 = conv(n_feat * 2, n_feat, kernel_size, bias=bias)
        self.concat23 = conv(n_feat * 2, n_feat + scale_orsnetfeats, kernel_size, bias=bias)
        # self.tail = conv(n_feat + scale_orsnetfeats, out_c, kernel_size, bias=bias)
        self.tail = conv(n_feat, out_c, kernel_size, bias=bias)

    def forward(self, x3_img):
        # Original-resolution Image for Stage 3
        H = x3_img.size(2)  # 128   x3_img (1,3,128,128)
        W = x3_img.size(3)  # 128

        # Multi-Patch Hierarchy: Split Image into four non-overlapping patches

        # Two Patches for Stage 2
        x2top_img = x3_img[:, :, 0:int(H / 2), :]  # 按H分割 (1,3,64,128)
        x2bot_img = x3_img[:, :, int(H / 2):H, :]

        # Four Patches for Stage 1
        x1ltop_img = x2top_img[:, :, :, 0:int(W / 2)]  # (1,3,64,64)
        x1rtop_img = x2top_img[:, :, :, int(W / 2):W]
        x1lbot_img = x2bot_img[:, :, :, 0:int(W / 2)]
        x1rbot_img = x2bot_img[:, :, :, int(W / 2):W]

        ##-------------------------------------------
        ##-------------- Stage 1---------------------
        ##-------------------------------------------
        ## Compute Shallow Features
        x1ltop = self.shallow_feat1(x1ltop_img)  # conv+CAB (1,80,64,64)
        x1rtop = self.shallow_feat1(x1rtop_img)
        x1lbot = self.shallow_feat1(x1lbot_img)
        x1rbot = self.shallow_feat1(x1rbot_img)

        ## Process features of all 4 patches with Encoder of Stage 1
        feat1_ltop = self.stage1_encoder(
            x1ltop)  # 左上块四个编码器从第一层到第四层的输出 (1,80,64,64) (1,128,32,32) (1,176,16,16) (1,224,8,8)
        feat1_rtop = self.stage1_encoder(x1rtop)
        feat1_lbot = self.stage1_encoder(x1lbot)
        feat1_rbot = self.stage1_encoder(x1rbot)

        ## Concat deep features
        feat1_top = [torch.cat((k, v), 3) for k, v in
                     zip(feat1_ltop, feat1_rtop)]  # (1,80,64,128) (1,128,32,64) (1,176,16,32) (1,224,8,16)
        feat1_bot = [torch.cat((k, v), 3) for k, v in zip(feat1_lbot, feat1_rbot)]

        ## Pass features through Decoder of Stage 1
        res1_top = self.stage1_decoder(feat1_top)  # (1,80,64,128) (1,128,32,64) (1,176,16,32)
        res1_bot = self.stage1_decoder(feat1_bot)

        ## Apply Supervised Attention Module (SAM)
        x2top_samfeats, stage1_img_top = self.sam12(res1_top[0], x2top_img)  # (1,80,64,128) (1,3,64,128)
        x2bot_samfeats, stage1_img_bot = self.sam12(res1_bot[0], x2bot_img)

        ## Output image at Stage 1
        stage1_img = torch.cat([stage1_img_top, stage1_img_bot], 2)  # (1，3，128，128)
        ##-------------------------------------------
        ##-------------- Stage 2---------------------
        ##-------------------------------------------
        ## Compute Shallow Features
        x2top = self.shallow_feat2(x2top_img)  # (1,80,64,128)
        x2bot = self.shallow_feat2(x2bot_img)  # (1,80,64,128)

        ## Concatenate SAM features of Stage 1 with shallow features of Stage 2
        x2top_cat = self.concat12(torch.cat([x2top, x2top_samfeats], 1))  # (1,80,64,128)
        x2bot_cat = self.concat12(torch.cat([x2bot, x2bot_samfeats], 1))

        ## Process features of both patches with Encoder of Stage 2
        feat2_top = self.stage2_encoder(x2top_cat, feat1_top, res1_top)  # (1,80,64,128) (1,128,32,64) (1,176,16,32) (1,224,8,16)
        feat2_bot = self.stage2_encoder(x2bot_cat, feat1_bot, res1_bot)

        ## Concat deep features
        feat2 = [torch.cat((k, v), 2) for k, v in
                 zip(feat2_top, feat2_bot)]  # (1,80,128,128) (1,128,64,64) (1,176,32,32) (1,224,16,16)

        ## Pass features through Decoder of Stage 2
        res2 = self.stage2_decoder(feat2)  # (1,80,128,128) (1,128,64,64) (1,176,32,32)

        ## Apply SAM
        x3_samfeats, stage2_img = self.sam23(res2[0], x3_img)  # (1,80,128,128) (1,3,128,128)

        ##-------------------------------------------
        ##-------------- Stage 3---------------------
        ##-------------------------------------------
        ## Compute Shallow Features
        x3 = self.ComplexNet(x3_img.type(torch.complex64))

        ## Concatenate SAM features of Stage 2 with shallow features of Stage 3
        # x3_cat = self.concat23(torch.cat([x3, x3_samfeats], 1))
        x3_cat = self.concat12(torch.cat([x3, x3_samfeats], 1))

        # x3_cat = self.stage3_orsnet(x3_cat, feat2, res2)
        x3_cat = self.stage3_encoder(x3_cat, feat2, res2)
        x3_cat = self.stage3_decoder(x3_cat)

        # stage3_img = self.tail(x3_cat)
        stage3_img = self.tail(x3_cat[0])

        return [stage3_img + x3_img, stage2_img, stage1_img]
