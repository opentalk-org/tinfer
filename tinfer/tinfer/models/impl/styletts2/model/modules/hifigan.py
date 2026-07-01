import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import remove_weight_norm, weight_norm

from .decoder_blocks import AdaINResBlock1, DecoderBackbone, SourceModuleHnNSF
from .utils import init_weights

LRELU_SLOPE = 0.1

class Generator(torch.nn.Module):
    def __init__(self, style_dim, resblock_kernel_sizes, upsample_rates, upsample_initial_channel, resblock_dilation_sizes, upsample_kernel_sizes):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        resblock = AdaINResBlock1

        self.m_source = SourceModuleHnNSF(
                    sampling_rate=24000,
                    upsample_scale=np.prod(upsample_rates),
                    harmonic_num=8, voiced_threshod=10)

        self.f0_upsamp = torch.nn.Upsample(scale_factor=np.prod(upsample_rates))
        self.noise_convs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.noise_res = nn.ModuleList()

        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            c_cur = upsample_initial_channel // (2 ** (i + 1))
            
            self.ups.append(weight_norm(ConvTranspose1d(upsample_initial_channel//(2**i), 
                         upsample_initial_channel//(2**(i+1)),
                         k, u, padding=(u//2 + u%2), output_padding=u%2)))
            
            if i + 1 < len(upsample_rates):  #
                stride_f0 = np.prod(upsample_rates[i + 1:])
                self.noise_convs.append(Conv1d(
                    1, c_cur, kernel_size=stride_f0 * 2, stride=stride_f0, padding=(stride_f0+1) // 2))
                self.noise_res.append(resblock(c_cur, 7, [1,3,5], style_dim))
            else:
                self.noise_convs.append(Conv1d(1, c_cur, kernel_size=1))
                self.noise_res.append(resblock(c_cur, 11, [1,3,5], style_dim))
            
        self.resblocks = nn.ModuleList()
        
        self.alphas = nn.ParameterList()
        self.alphas.append(nn.Parameter(torch.ones(1, upsample_initial_channel, 1)))
        
        for i in range(len(self.ups)):
            ch = upsample_initial_channel//(2**(i+1))
            self.alphas.append(nn.Parameter(torch.ones(1, ch, 1)))
            
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d, style_dim))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def _preprocess_f0(self, f0):
        f0 = self.f0_upsamp(f0[:, None]).transpose(1, 2)  # bs,n,t
        har_source, noi_source, uv = self.m_source(f0)
        return har_source.transpose(1, 2)

    def _pre_upsample_noise(self, x, har_source, s, i):
        x = x + (1 / self.alphas[i]) * (torch.sin(self.alphas[i] * x) ** 2)
        x_source = self.noise_convs[i](har_source)
        return x, self.noise_res[i](x_source, s)

    def _process_resblocks(self, x, s, i):
        xs = None
        for j in range(self.num_kernels):
            res_out = self.resblocks[i * self.num_kernels + j](x, s)
            if xs is None:
                xs = res_out.clone() if hasattr(res_out, "clone") else res_out
            else:
                xs = xs + res_out
        return xs / self.num_kernels

    def _generate_output(self, x):
        x = x + (1 / self.alphas[-1]) * (torch.sin(self.alphas[-1] * x) ** 2)
        x = self.conv_post(x)
        return torch.tanh(x)

    def _forward_compiled(self, x, s, har_source):
        for i in range(self.num_upsamples):
            x, x_source = self._pre_upsample_noise(x, har_source, s, i)
            x = self.ups[i](x)
            x = x + x_source
            x = self._process_resblocks(x, s, i)
        return self._generate_output(x)

    def forward(self, x, s, f0):
        har_source = self._preprocess_f0(f0)
        return self._forward_compiled(x, s, har_source)

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)

        
class Decoder(DecoderBackbone):
    def __init__(self, dim_in=512, F0_channel=512, style_dim=64, dim_out=80, 
                resblock_kernel_sizes = [3,7,11],
                upsample_rates = [10,5,3,2],
                upsample_initial_channel=512,
                resblock_dilation_sizes=[[1,3,5], [1,3,5], [1,3,5]],
                upsample_kernel_sizes=[20,10,6,4]):
        super().__init__(dim_in=dim_in, style_dim=style_dim)
        self.generator = Generator(style_dim, resblock_kernel_sizes, upsample_rates, upsample_initial_channel, resblock_dilation_sizes, upsample_kernel_sizes)
        
    def forward(self, asr, F0_curve, N, s):
        x, F0_curve = super().forward(asr, F0_curve, N, s)
        x = self.generator(x, s, F0_curve)
        return x