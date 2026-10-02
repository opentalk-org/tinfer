from collections import OrderedDict

import torch
import torch.nn as nn
import yaml
from munch import Munch

from .blocks.text import TextEncoder
from .blocks.prosody import ProsodyPredictor
from .blocks.style import StyleEncoder
from .config import ModelConfig, TrainingArgs, convert_style_tts2_config
from .diffusion.diffusion import AudioDiffusionConditional
from .diffusion.networks import Transformer1d, StyleTransformer1d
from .diffusion.sampling import KDiffusion, LogNormalDistribution
from .hifigan import Decoder as HifiganDecoder
from .istftnet import Decoder as IstftnetDecoder
from .plbert import build_plbert

def build_model(model_config: ModelConfig, build_style_encoder: bool = True):

    assert model_config.decoder.type in ['istftnet', 'hifigan'], 'Decoder type unknown'

    if model_config.decoder.type == "istftnet":
        decoder = IstftnetDecoder(dim_in=model_config.hidden_dim, style_dim=model_config.style_dim, dim_out=model_config.n_mels,
                resblock_kernel_sizes = model_config.decoder.resblock_kernel_sizes,
                upsample_rates = model_config.decoder.upsample_rates,
                upsample_initial_channel=model_config.decoder.upsample_initial_channel,
                resblock_dilation_sizes=model_config.decoder.resblock_dilation_sizes,
                upsample_kernel_sizes=model_config.decoder.upsample_kernel_sizes,
                gen_istft_n_fft=model_config.decoder.gen_istft_n_fft, gen_istft_hop_size=model_config.decoder.gen_istft_hop_size, grad_checkpoint=False)
    else:
        decoder = HifiganDecoder(dim_in=model_config.hidden_dim, style_dim=model_config.style_dim, dim_out=model_config.n_mels,
                resblock_kernel_sizes = model_config.decoder.resblock_kernel_sizes,
                upsample_rates = model_config.decoder.upsample_rates,
                upsample_initial_channel=model_config.decoder.upsample_initial_channel,
                resblock_dilation_sizes=model_config.decoder.resblock_dilation_sizes,
                upsample_kernel_sizes=model_config.decoder.upsample_kernel_sizes)

    text_encoder = TextEncoder(channels=model_config.hidden_dim, kernel_size=5, depth=model_config.n_layer, n_symbols=model_config.n_token)

    predictor = ProsodyPredictor(style_dim=model_config.style_dim, d_hid=model_config.hidden_dim, nlayers=model_config.n_layer, max_dur=model_config.max_dur, dropout=model_config.dropout)

    bert = build_plbert(model_config.plbert_config)

    transformer_dict = {
        'num_layers': model_config.diffusion.transformer.num_layers,
        'num_heads': model_config.diffusion.transformer.num_heads,
        'head_features': model_config.diffusion.transformer.head_features,
        'multiplier': model_config.diffusion.transformer.multiplier,
    }

    if model_config.multispeaker:
        transformer = StyleTransformer1d(channels=model_config.style_dim*2,
                                    context_embedding_features=bert.config.hidden_size,
                                    context_features=model_config.style_dim*2,
                                    **transformer_dict)
    else:
        transformer = Transformer1d(channels=model_config.style_dim*2,
                                    context_embedding_features=bert.config.hidden_size,
                                    **transformer_dict)

    diffusion = AudioDiffusionConditional(
        in_channels=1,
        embedding_max_length=bert.config.max_position_embeddings,
        embedding_features=bert.config.hidden_size,
        embedding_mask_proba=model_config.diffusion.embedding_mask_proba,
        channels=model_config.style_dim*2,
        context_features=model_config.style_dim*2,
    )

    diffusion.diffusion = KDiffusion(
        net=diffusion.unet,
        sigma_distribution=LogNormalDistribution(mean = model_config.diffusion.dist.mean, std = model_config.diffusion.dist.std),
        sigma_data=model_config.diffusion.dist.sigma_data,
        dynamic_threshold=0.0
    )
    diffusion.diffusion.net = transformer
    diffusion.unet = transformer

    nets = Munch(
            bert=bert,
            bert_encoder=nn.Linear(bert.config.hidden_size, model_config.hidden_dim),

            predictor=predictor,
            decoder=decoder,
            text_encoder=text_encoder,

            diffusion=diffusion,
       )

    if build_style_encoder:
        nets.style_encoder = StyleEncoder(dim_in=model_config.dim_in, style_dim=model_config.style_dim, max_conv_dim=model_config.hidden_dim, max_length=model_config.max_style_length)
        nets.predictor_encoder = StyleEncoder(dim_in=model_config.dim_in, style_dim=model_config.style_dim, max_conv_dim=model_config.hidden_dim, max_length=model_config.max_style_length)

    return nets

def load_original_styletts2_config(config_path: str) -> TrainingArgs:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return convert_style_tts2_config(config)

def _strip_module_prefix(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            name = k[7:]
        else:
            name = k
        new_state_dict[name] = v
    return new_state_dict

def load_original_styletts2_model(model_path: str, config_path: str):
    model_saved = torch.load(model_path, map_location='cpu', weights_only=True)

    config = load_original_styletts2_config(config_path)

    model_config = config.model_params
    model_state_dict = model_saved['net']

    model = build_model(model_config, True)

    for key in model.keys():
        if key in model_state_dict:
            state_dict = model_state_dict[key]
            state_dict = _strip_module_prefix(state_dict)
            model[key].load_state_dict(state_dict, strict=False)
        else:
            raise ValueError(f"Key {key} not found in model state dict")

    _ = [model[key].eval() for key in model]

    return model, model_config
