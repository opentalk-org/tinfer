from .istftnet import Decoder as IstftnetDecoder
from .hifigan import Decoder as HifiganDecoder
import torch.nn as nn
import torch
from .blocks import TextEncoder, ProsodyPredictor, StyleEncoder
from .diffusion.sampler import KDiffusion, LogNormalDistribution
from .diffusion.modules import Transformer1d, StyleTransformer1d
from .diffusion.diffusion import AudioDiffusionConditional
from .discriminators import MultiPeriodDiscriminator, MultiResSpecDiscriminator, WavLMDiscriminator
from tinfer.models.impl.styletts2.model.modules.istftnet import TorchSTFT
from munch import Munch
from .asr.models import ASRCNN
from .jdc.model import JDCNet
from .config import ModelConfig, ASRConfig, Stage, TrainingArgs, convert_style_tts2_config, DecoderConfig, DiffusionConfig, DiffusionTransformerConfig, DiffusionDistributionConfig, PreprocessConfig
from .plbert import build_plbert
import yaml
from collections import OrderedDict
from dataclasses import asdict

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

def _parse_model_config(config_dict: dict) -> ModelConfig:
    if isinstance(config_dict, ModelConfig):
        return config_dict
    elif isinstance(config_dict, dict):
        config_dict = config_dict.copy()
        if 'decoder' in config_dict and isinstance(config_dict['decoder'], dict):
            config_dict['decoder'] = DecoderConfig(**config_dict['decoder'])
        if 'diffusion' in config_dict and isinstance(config_dict['diffusion'], dict):
            diffusion_dict = config_dict['diffusion'].copy()
            if 'transformer' in diffusion_dict and isinstance(diffusion_dict['transformer'], dict):
                diffusion_dict['transformer'] = DiffusionTransformerConfig(**diffusion_dict['transformer'])
            if 'dist' in diffusion_dict and isinstance(diffusion_dict['dist'], dict):
                diffusion_dict['dist'] = DiffusionDistributionConfig(**diffusion_dict['dist'])
            config_dict['diffusion'] = DiffusionConfig(**diffusion_dict)
        if 'preprocess' in config_dict and isinstance(config_dict['preprocess'], dict):
            config_dict['preprocess'] = PreprocessConfig(**config_dict['preprocess'])
        return ModelConfig(**config_dict)
    else:
        raise ValueError(f"Unexpected config type: {type(config_dict)}")

def load_model_from_state(state_dict: dict, load_style_encoder: bool = True):
    config_dict = state_dict['config']
    model_config = _parse_model_config(config_dict)
    model_state_dict = state_dict['net']

    model = build_model(model_config, load_style_encoder)
    
    for key in model.keys():
        if key in model_state_dict:
            if key == "decoder" and "generator.stft.window" not in model_state_dict[key]:
                # TODO: move this to model conversion
                gen_istft_n_fft = 20
                gen_istft_hop_size = 5
                stft = TorchSTFT(
                    filter_length=gen_istft_n_fft,
                    hop_length=gen_istft_hop_size,
                    win_length=gen_istft_n_fft
                )
                model_state_dict[key]["generator.stft.window"] = stft.window
            model[key].load_state_dict(model_state_dict[key])
        else:
            raise ValueError(f"Key {key} not found in model state dict")
    
    _ = [model[key].eval() for key in model]

    return model, model_config

def load_model(model_path: str, load_style_encoder: bool = True):
    model_saved = torch.load(model_path, map_location='cpu', weights_only=True)
    return load_model_from_state(model_saved, load_style_encoder)

def get_model_state_dict(model: nn.Module, config: ModelConfig, runtime_config: dict | None = None) -> dict:
    config_dict = asdict(config)
    state_dict = {
        'config': config_dict,
        'net': {key: model[key].state_dict() for key in model},
    }
    if runtime_config is not None:
        state_dict['runtime_config'] = runtime_config
    return state_dict

def save_model(model: nn.Module, config: ModelConfig, model_path: str, runtime_config: dict | None = None):
    save_dict = get_model_state_dict(model, config, runtime_config)
    torch.save(save_dict, model_path)

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

def load_original_styletts2_checkpoint(checkpoint_path: str, config_path: str):
    pass
    
    
def build_training_model(args: TrainingArgs, stage: Stage):
    
    if args.grad_checkpoint_generator and args.model_params.decoder.type == "hifigan":
        raise ValueError("Using grad checkpoint for hifigan decoder is not supported!")

    nets = build_model(args.model_params, None)

    if args.grad_checkpoint_generator is True:
        nets.decoder.grad_checkpoint = True

    if args.ASR_config is not None and args.ASR_path is not None:
        nets.text_aligner = load_ASR_models(args.ASR_path, args.ASR_config)
    elif args.ASR_config is not None:
        nets.text_aligner = build_ASR_model(args.ASR_config)
    else:
        raise ValueError("ASR_config is required")

    if args.F0_path is not None:
        nets.pitch_extractor = load_F0_models(args.F0_path)
    else:
        raise ValueError("F0_path is required")

    if stage != Stage.FIRST:
        nets.mpd = MultiPeriodDiscriminator(grad_checkpoint=args.grad_checkpoint_gans)
        nets.msd = MultiResSpecDiscriminator(grad_checkpoint=args.grad_checkpoint_gans)
        nets.wd = WavLMDiscriminator(args.slm.hidden, args.slm.nlayers, args.slm.initial_channel)

    return nets

def load_F0_models(path):
    F0_model = JDCNet(num_class=1, seq_len=192)
    params = torch.load(path, map_location='cpu')['net']
    F0_model.load_state_dict(params)
    _ = F0_model.train()
    
    return F0_model

def build_ASR_model(ASR_MODEL_CONFIG):
    with open(ASR_MODEL_CONFIG, 'r') as f:
        config_data = yaml.safe_load(f)
    if 'model_params' in config_data:
        asr_config = ASRConfig(**config_data['model_params'])
    else:
        asr_config = ASRConfig(**config_data)
    model = ASRCNN(
        input_dim=asr_config.input_dim,
        hidden_dim=asr_config.hidden_dim,
        n_token=asr_config.n_token,
        n_layers=asr_config.n_layers,
        token_embedding_dim=asr_config.token_embedding_dim
    )
    return model

def load_ASR_models(ASR_MODEL_PATH, ASR_MODEL_CONFIG):
    model = build_ASR_model(ASR_MODEL_CONFIG)
    params = torch.load(ASR_MODEL_PATH, map_location='cpu', weights_only=False)['model']
    model.load_state_dict(params)
    _ = model.train()
    return model

