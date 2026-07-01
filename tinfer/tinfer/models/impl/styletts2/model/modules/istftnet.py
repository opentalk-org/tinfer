import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import os
from scipy.signal import get_window
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import remove_weight_norm, weight_norm

from .decoder_blocks import AdaINResBlock1, DecoderBackbone, SourceModuleHnNSF
from .utils import init_weights

LRELU_SLOPE = 0.1

class TorchSTFT(torch.nn.Module):
    def __init__(self, filter_length=800, hop_length=200, win_length=800, window='hann'):
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        window_tensor = torch.from_numpy(get_window(window, win_length, fftbins=True).astype(np.float32))
        self.register_buffer('window', window_tensor)

    def transform(self, input_data):
        forward_transform = torch.stft(
            input_data,
            self.filter_length, self.hop_length, self.win_length, window=self.window,
            return_complex=True)

        return torch.abs(forward_transform), torch.angle(forward_transform)

    def inverse(self, magnitude, phase):
        if (
            os.getenv("TINFER_TRT_EXPORT") == "1"
            and self.filter_length == 20
            and self.hop_length == 5
            and self.win_length == 20
        ):
            from .tensorrt_export import onnx_istft20_inverse

            return onnx_istft20_inverse(magnitude, phase, self.window)

        inverse_transform = torch.istft(
            magnitude * (torch.cos(phase) + 1j * torch.sin(phase)),
            self.filter_length, self.hop_length, self.win_length, window=self.window)

        return inverse_transform.unsqueeze(-2)  # unsqueeze to stay consistent with conv_transpose1d implementation

    def forward(self, input_data):
        self.magnitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction
    
class Generator(torch.nn.Module):
    def __init__(self, style_dim, resblock_kernel_sizes, upsample_rates, upsample_initial_channel, resblock_dilation_sizes, upsample_kernel_sizes, gen_istft_n_fft, gen_istft_hop_size, grad_checkpoint=False):
        super(Generator, self).__init__()

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        resblock = AdaINResBlock1

        upsample_scale = int(np.prod(upsample_rates) * gen_istft_hop_size)
        self.m_source = SourceModuleHnNSF(
                    sampling_rate=24000,
                    upsample_scale=upsample_scale,
                    harmonic_num=8, voiced_threshod=10)
        self.f0_upsamp = torch.nn.Upsample(scale_factor=float(upsample_scale))
        self.noise_convs = nn.ModuleList()
        self.noise_res = nn.ModuleList()
        
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(upsample_initial_channel//(2**i), upsample_initial_channel//(2**(i+1)),
                                k, u, padding=(k-u)//2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes,resblock_dilation_sizes)):
                block = resblock(ch, k, d, style_dim)
                self.resblocks.append(block)
                
            c_cur = upsample_initial_channel // (2 ** (i + 1))
            
            if i + 1 < len(upsample_rates):  #
                stride_f0 = int(np.prod(upsample_rates[i + 1:]))
                self.noise_convs.append(Conv1d(
                    gen_istft_n_fft + 2, c_cur, kernel_size=stride_f0 * 2, stride=stride_f0, padding=(stride_f0+1) // 2))
                noise_block = resblock(c_cur, 7, [1,3,5], style_dim)
                self.noise_res.append(noise_block)
            else:
                self.noise_convs.append(Conv1d(gen_istft_n_fft + 2, c_cur, kernel_size=1))
                noise_block = resblock(c_cur, 11, [1,3,5], style_dim)
                self.noise_res.append(noise_block)
                
                
        self.post_n_fft = gen_istft_n_fft
        self.conv_post = weight_norm(Conv1d(ch, self.post_n_fft + 2, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.reflection_pad = torch.nn.ReflectionPad1d((1, 0))
        self.stft = TorchSTFT(filter_length=gen_istft_n_fft, hop_length=gen_istft_hop_size, win_length=gen_istft_n_fft)
    
    def _preprocess_f0(self, f0):
        with torch.no_grad():
            f0 = self.f0_upsamp(f0[:, None]).transpose(1, 2)  # bs,n,t

            har_source, noi_source, uv = self.m_source(f0)
            har_source = har_source.transpose(1, 2).squeeze(1)
            har_spec, har_phase = self.stft.transform(har_source)
            har = torch.cat([har_spec, har_phase], dim=1)
        return har

    def _forward_compiled(self, x, s, har):
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x_source = self._noise_conv_block(har, s, i)
            
            x = self.ups[i](x)
            if i == self.num_upsamples - 1:
                x = self.reflection_pad(x)
                
            # x = x + x_source
            x += x_source
            x = self._process_resblocks(x, s, i)
        return self._generate_output(x)
        
    
    def forward(self, x, s, f0):
        har = self._preprocess_f0(f0)
        return self._forward_compiled(x, s, har)

    def _noise_conv_block(self, har, s, i):
        x_source = self.noise_convs[i](har)
        return self.noise_res[i](x_source, s)

    def _generate_output(self, x):
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        spec = torch.exp(x[:,:self.post_n_fft // 2 + 1, :])
        phase = torch.sin(x[:, self.post_n_fft // 2 + 1:, :])
        return self.stft.inverse(spec, phase)

    def _process_resblocks(self, x, s, i):
        xs = None
        for j in range(self.num_kernels):
            res_out = self.resblocks[i*self.num_kernels+j](x, s)
            if xs is None:
                xs = res_out.clone() if hasattr(res_out, 'clone') else res_out
            else:
                xs = xs + res_out
        return xs / self.num_kernels
        
    
    def fw_phase(self, x, s):
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x, s)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x, s)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.reflection_pad(x)
        x = self.conv_post(x)
        spec = torch.exp(x[:,:self.post_n_fft // 2 + 1, :])
        phase = torch.sin(x[:, self.post_n_fft // 2 + 1:, :])
        return spec, phase

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
                upsample_rates = [10, 6],
                upsample_initial_channel=512,
                resblock_dilation_sizes=[[1,3,5], [1,3,5], [1,3,5]],
                upsample_kernel_sizes=[20, 12], 
                gen_istft_n_fft=20, gen_istft_hop_size=5, grad_checkpoint=False):
        super().__init__(dim_in=dim_in, style_dim=style_dim)
        self.generator = Generator(style_dim, resblock_kernel_sizes, upsample_rates, 
                                   upsample_initial_channel, resblock_dilation_sizes, 
                                   upsample_kernel_sizes, gen_istft_n_fft, gen_istft_hop_size, grad_checkpoint)
        
    def forward(self, asr, F0_curve, N, s):
        x, F0_curve = super().forward(asr, F0_curve, N, s)
        x = self.generator(x, s, F0_curve)
        return x

    def forward_with_har(self, asr, F0_curve, N, s, har):
        x, _ = super().forward(asr, F0_curve, N, s)
        return self.generator._forward_compiled(x, s, har)
