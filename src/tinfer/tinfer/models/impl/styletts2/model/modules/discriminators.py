import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, AvgPool1d, Conv2d
from torch.nn.utils.parametrizations import weight_norm, spectral_norm


from .utils import get_padding

LRELU_SLOPE = 0.1

def stft(x, fft_size, hop_size, win_length, window):
    """Perform STFT and convert to magnitude spectrogram.
    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.
    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
    """
    x_stft = torch.stft(x, fft_size, hop_size, win_length, window,
            return_complex=True)
    real = x_stft[..., 0]
    imag = x_stft[..., 1]

    return torch.abs(x_stft).transpose(2, 1)

class SpecDiscriminator(nn.Module):
    """docstring for Discriminator."""

    def __init__(self, fft_size=1024, shift_size=120, win_length=600, window="hann_window", use_spectral_norm=False, grad_checkpoint=False):
        super(SpecDiscriminator, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.window = getattr(torch, window)(win_length)
        self.discriminators = nn.ModuleList([
            norm_f(nn.Conv2d(1, 32, kernel_size=(3, 9), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1,2), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1,2), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1,2), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1,1), padding=(1, 1))),
        ])

        self.out = norm_f(nn.Conv2d(32, 1, 3, 1, 1))
        self.dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)
        self.grad_checkpoint = grad_checkpoint
    def forward(self, y, compute_loss=False, fmap_g=None):
        fmap = []
        y = y.squeeze(1)
        
        y = stft(y, self.fft_size, self.shift_size, self.win_length, self.window.to(y.device))
        y = y.unsqueeze(1)
        
        feature_loss_val = 0.0
        
        for i, d in enumerate(self.discriminators):
            if self.grad_checkpoint:
                y = torch.utils.checkpoint.checkpoint(self._discriminator_block, y, i, self.dummy_tensor, use_reentrant=False)
            else:
                y = self._discriminator_block(y, i, self.dummy_tensor)

            if compute_loss and fmap_g is not None:
                feature_loss_val += torch.mean(torch.abs(y - fmap_g[i]))
            elif not compute_loss:
                fmap.append(y)

        y = self.out(y)
        if compute_loss and fmap_g is not None:
            feature_loss_val += torch.mean(torch.abs(y - fmap_g[-1]))
        elif not compute_loss:
            fmap.append(y)

        output = torch.flatten(y, 1, -1)
        
        if compute_loss:
            return output, feature_loss_val * 2
        else:
            return output, fmap

    def _discriminator_block(self, x, i, dummy_tensor):
        d = self.discriminators[i]
        return F.leaky_relu(d(x), LRELU_SLOPE)
class MultiResSpecDiscriminator(torch.nn.Module):

    def __init__(self,
                 fft_sizes=[1024, 2048, 512],
                 hop_sizes=[120, 240, 50],
                 win_lengths=[600, 1200, 240],
                 window="hann_window",
                 grad_checkpoint=False):

        super(MultiResSpecDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            SpecDiscriminator(fft_sizes[0], hop_sizes[0], win_lengths[0], window, grad_checkpoint=grad_checkpoint),
            SpecDiscriminator(fft_sizes[1], hop_sizes[1], win_lengths[1], window, grad_checkpoint=grad_checkpoint),
            SpecDiscriminator(fft_sizes[2], hop_sizes[2], win_lengths[2], window, grad_checkpoint=grad_checkpoint)
            ])

    def forward(self, y, y_hat, compute_loss=False):
        y_d_rs = []
        y_d_gs = []
        feature_loss_val = 0.0
        
        if compute_loss:
            for i, d in enumerate(self.discriminators):
                y_d_r, fmap_r = d(y, compute_loss=False)
                y_d_rs.append(y_d_r)
                y_d_g, feature_loss = d(y_hat, compute_loss=True, fmap_g=fmap_r)
                y_d_gs.append(y_d_g)
                feature_loss_val += feature_loss
                del fmap_r
        else:
            for i, d in enumerate(self.discriminators):
                y_d_r, _ = d(y, compute_loss=False)
                y_d_g, _ = d(y_hat, compute_loss=False)
                y_d_rs.append(y_d_r)
                y_d_gs.append(y_d_g)

        if compute_loss:
            return y_d_rs, y_d_gs, feature_loss_val
        else:
            return y_d_rs, y_d_gs, None


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False, grad_checkpoint=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))
        self.dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)
        self.grad_checkpoint = grad_checkpoint

    def forward(self, x, compute_loss=False, fmap_g=None):
        fmap = []

        if not self.dummy_tensor.requires_grad:
            self.dummy_tensor.requires_grad = True

        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        feature_loss_val = 0.0
        
        for i in range(len(self.convs)):
            if self.grad_checkpoint:
                x = torch.utils.checkpoint.checkpoint(self.layer_with_relu, x, i, self.dummy_tensor, use_reentrant=False)
            else:
                x = self.layer_with_relu(x, i, self.dummy_tensor)
            if compute_loss and fmap_g is not None:
                feature_loss_val += torch.mean(torch.abs(x - fmap_g[i]))
            elif not compute_loss:
                fmap.append(x)

        x = self.conv_post(x)
        if compute_loss and fmap_g is not None:
            feature_loss_val += torch.mean(torch.abs(x - fmap_g[-1]))
        elif not compute_loss:
            fmap.append(x)
        x = torch.flatten(x, 1, -1)

        if compute_loss:
            return x, feature_loss_val * 2
        else:
            return x, fmap

    def layer_with_relu(self, x, i, dummy_tensor):
        return F.leaky_relu(self.convs[i](x), LRELU_SLOPE)

class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, grad_checkpoint=False):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(2, grad_checkpoint=grad_checkpoint),
            DiscriminatorP(3, grad_checkpoint=grad_checkpoint),
            DiscriminatorP(5, grad_checkpoint=grad_checkpoint),
            DiscriminatorP(7, grad_checkpoint=grad_checkpoint),
            DiscriminatorP(11, grad_checkpoint=grad_checkpoint),
        ])

    def forward(self, y, y_hat, compute_loss=False):
        y_d_rs = []
        y_d_gs = []
        feature_loss_val = 0.0
        
        if compute_loss:
            for i, d in enumerate(self.discriminators):
                y_d_r, fmap_r = d(y, compute_loss=False)
                y_d_rs.append(y_d_r)
                y_d_g, feature_loss = d(y_hat, compute_loss=True, fmap_g=fmap_r)
                y_d_gs.append(y_d_g)
                feature_loss_val += feature_loss
                del fmap_r
        else:
            for i, d in enumerate(self.discriminators):
                y_d_r, _ = d(y, compute_loss=False)
                y_d_g, _ = d(y_hat, compute_loss=False)
                y_d_rs.append(y_d_r)
                y_d_gs.append(y_d_g)

        if compute_loss:
            return y_d_rs, y_d_gs, feature_loss_val
        else:
            return y_d_rs, y_d_gs, None
    
class WavLMDiscriminator(nn.Module):
    """docstring for Discriminator."""

    def __init__(self, slm_hidden=768, 
                 slm_layers=13, 
                 initial_channel=64, 
                 use_spectral_norm=False):
        super(WavLMDiscriminator, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.pre = norm_f(Conv1d(slm_hidden * slm_layers, initial_channel, 1, 1, padding=0))
        
        self.convs = nn.ModuleList([
            norm_f(nn.Conv1d(initial_channel, initial_channel * 2, kernel_size=5, padding=2)),
            norm_f(nn.Conv1d(initial_channel * 2, initial_channel * 4, kernel_size=5, padding=2)),
            norm_f(nn.Conv1d(initial_channel * 4, initial_channel * 4, 5, 1, padding=2)),
        ])

        self.conv_post = norm_f(Conv1d(initial_channel * 4, 1, 3, 1, padding=1))
        
    def forward(self, x):
        x = self.pre(x)
        
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        x = torch.flatten(x, 1, -1)

        return x