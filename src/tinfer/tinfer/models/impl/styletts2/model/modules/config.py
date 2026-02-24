from dataclasses import dataclass, field
from typing import List, Tuple
import yaml
from enum import Enum

@dataclass
class PreprocessConfig:
    sr: int = 24000
    n_fft: int = 2048
    win_length: int = 1200
    hop_length: int = 300

@dataclass
class ASRConfig:
    input_dim: int = 80
    hidden_dim: int = 256
    n_token: int = 35
    n_layers: int = 6
    token_embedding_dim: int = 256

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ASRConfig":
        with open(yaml_path, 'r') as f:
            config_data = yaml.safe_load(f)
        if 'model_params' in config_data:
            return cls(**config_data['model_params'])
        return cls(**config_data)

@dataclass
class DataConfig:
    train_data: str = "data/processed/train_list.txt"
    val_data: str = "data/processed/val_list.txt"
    root_path: str = "data/processed/wavs"
    OOD_data: str = "data/processed/OOD_texts.txt"
    min_length: int = 50


@dataclass
class DecoderConfig:
    type: str = "istftnet"
    resblock_kernel_sizes: List[int] = field(default_factory=lambda: [3, 7, 11])
    upsample_rates: List[int] = field(default_factory=lambda: [10, 6])
    upsample_initial_channel: int = 512
    resblock_dilation_sizes: List[List[int]] = field(default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]])
    upsample_kernel_sizes: List[int] = field(default_factory=lambda: [20, 12])
    gen_istft_n_fft: int = 20
    gen_istft_hop_size: int = 5


@dataclass
class SLMConfig:
    model: str = "microsoft/wavlm-base-plus"
    sr: int = 16000
    hidden: int = 768
    nlayers: int = 13
    initial_channel: int = 64


@dataclass
class DiffusionTransformerConfig:
    num_layers: int = 3
    num_heads: int = 8
    head_features: int = 64
    multiplier: int = 2


@dataclass
class DiffusionDistributionConfig:
    sigma_data: float = 0.2
    estimate_sigma_data: bool = True
    mean: float = -3.0
    std: float = 1.0

@dataclass
class DiffusionConfig:
    embedding_mask_proba: float = 0.1
    transformer: DiffusionTransformerConfig = field(default_factory=DiffusionTransformerConfig)
    dist: DiffusionDistributionConfig = field(default_factory=DiffusionDistributionConfig)


@dataclass
class ModelConfig:
    multispeaker: bool = True
    max_style_length: int = 400
    dim_in: int = 64
    hidden_dim: int = 512
    max_conv_dim: int = 512
    n_layer: int = 3
    n_mels: int = 80
    n_token: int = 178
    max_dur: int = 50
    style_dim: int = 128
    dropout: float = 0.2

    plbert_config: dict = field(default_factory=dict)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)


@dataclass
class LossConfig:
    lambda_mel: float = 5.0
    lambda_gen: float = 1.0
    lambda_slm: float = 1.0
    lambda_mono: float = 1.0
    lambda_s2s: float = 1.0
    TMA_epoch: int = 0
    lambda_F0: float = 1.0
    lambda_norm: float = 1.0
    lambda_dur: float = 1.0
    lambda_ce: float = 20.0
    lambda_sty: float = 1.0
    lambda_diff: float = 1.0
    diff_epoch: int = 20
    joint_epoch: int = 30

@dataclass
class OptimizerConfig:
    lr: float = 0.0001
    max_lr: float = 0.0001
    bert_lr: float = 0.00001
    ft_lr: float = 0.00001
    pct_start: float = 0.0
    div_factor: float = 1.0
    final_div_factor: float = 1.0
    max_grad_norm: float = 10.0
    weight_decay: float = 1e-4
    betas: Tuple[float, float] = (0.0, 0.99)

@dataclass
class SLMAdvConfig:
    min_len: int = 400
    max_len: int = 500
    batch_percentage: float = 0.5
    iter: int = 10
    thresh: int = 5
    scale: float = 0.01
    sig: float = 1.5

@dataclass
class TrainingArgs:
    log_dir: str = "logs/tts"
    save_freq: int = 2
    audio_log_freq: int = 2
    log_interval: int = 10
    device: str = "cuda"
    
    batch_size: int = 16
    max_len: int = 400

    grad_checkpoint_gans: bool = False
    grad_checkpoint_generator: bool = False

    F0_path: str|None = None
    ASR_config: str|None = None
    ASR_path: str|None = None
    BERT_path: str|None = None

    data_params: DataConfig = field(default_factory=DataConfig)
    preprocess_params: PreprocessConfig = field(default_factory=PreprocessConfig)
    model_params: ModelConfig = field(default_factory=ModelConfig)
    loss_params: LossConfig = field(default_factory=LossConfig)
    optimizer_params: OptimizerConfig = field(default_factory=OptimizerConfig)

    slmadv_params: SLMAdvConfig = field(default_factory=SLMAdvConfig)
    slm: SLMConfig = field(default_factory=SLMConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "TrainingArgs":
        with open(yaml_path, 'r') as f:
            config_data = yaml.safe_load(f)
        return cls(**config_data)

    def to_yaml(self, yaml_path: str):
        from dataclasses import asdict
        config_data = asdict(self)
        with open(yaml_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

def convert_style_tts2_config(style_tts2_config: dict) -> TrainingArgs:
    import os
    
    def require_key(d: dict, key: str, path: str = ""):
        if key not in d:
            full_path = f"{path}.{key}" if path else key
            raise KeyError(f"Required config key '{full_path}' is missing")
        return d[key]
    
    converted = {}
    
    converted['log_dir'] = require_key(style_tts2_config, 'log_dir')
    converted['save_freq'] = require_key(style_tts2_config, 'save_freq')
    converted['log_interval'] = require_key(style_tts2_config, 'log_interval')
    converted['device'] = require_key(style_tts2_config, 'device')
    converted['batch_size'] = require_key(style_tts2_config, 'batch_size')
    converted['max_len'] = require_key(style_tts2_config, 'max_len')
    converted['F0_path'] = require_key(style_tts2_config, 'F0_path')
    converted['ASR_config'] = require_key(style_tts2_config, 'ASR_config')
    converted['ASR_path'] = require_key(style_tts2_config, 'ASR_path')
    plbert_dir = require_key(style_tts2_config, 'PLBERT_dir')
    converted['BERT_path'] = plbert_dir
    
    plbert_config = {}
    if plbert_dir:
        plbert_config_path = os.path.join(plbert_dir, 'config.yml')
        if not os.path.exists(plbert_config_path):
            raise FileNotFoundError(f"PLBERT config file not found: {plbert_config_path}")
        with open(plbert_config_path, 'r') as f:
            plbert_config_data = yaml.safe_load(f)
            if 'model_params' not in plbert_config_data:
                raise KeyError(f"Required key 'model_params' not found in PLBERT config: {plbert_config_path}")
            plbert_config = plbert_config_data['model_params']
    
    data_dict = require_key(style_tts2_config, 'data_params')
    converted['data_params'] = DataConfig(
        train_data=require_key(data_dict, 'train_data', 'data_params'),
        val_data=require_key(data_dict, 'val_data', 'data_params'),
        root_path=require_key(data_dict, 'root_path', 'data_params'),
        OOD_data=require_key(data_dict, 'OOD_data', 'data_params'),
        min_length=require_key(data_dict, 'min_length', 'data_params')
    )
    
    preprocess_dict = require_key(style_tts2_config, 'preprocess_params')
    spect_params = require_key(preprocess_dict, 'spect_params', 'preprocess_params')
    converted['preprocess_params'] = PreprocessConfig(
        sr=require_key(preprocess_dict, 'sr', 'preprocess_params'),
        n_fft=require_key(spect_params, 'n_fft', 'preprocess_params.spect_params'),
        win_length=require_key(spect_params, 'win_length', 'preprocess_params.spect_params'),
        hop_length=require_key(spect_params, 'hop_length', 'preprocess_params.spect_params')
    )
    
    model_dict = require_key(style_tts2_config, 'model_params')
    
    decoder_dict = require_key(model_dict, 'decoder', 'model_params')
    _default_decoder = DecoderConfig()
    decoder_config = DecoderConfig(
        type=require_key(decoder_dict, 'type', 'model_params.decoder'),
        resblock_kernel_sizes=require_key(decoder_dict, 'resblock_kernel_sizes', 'model_params.decoder'),
        upsample_rates=require_key(decoder_dict, 'upsample_rates', 'model_params.decoder'),
        upsample_initial_channel=require_key(decoder_dict, 'upsample_initial_channel', 'model_params.decoder'),
        resblock_dilation_sizes=require_key(decoder_dict, 'resblock_dilation_sizes', 'model_params.decoder'),
        upsample_kernel_sizes=require_key(decoder_dict, 'upsample_kernel_sizes', 'model_params.decoder'),
        gen_istft_n_fft=decoder_dict.get('gen_istft_n_fft', _default_decoder.gen_istft_n_fft),
        gen_istft_hop_size=decoder_dict.get('gen_istft_hop_size', _default_decoder.gen_istft_hop_size)
    )
    
    diffusion_dict = require_key(model_dict, 'diffusion', 'model_params')
    transformer_dict = require_key(diffusion_dict, 'transformer', 'model_params.diffusion')
    dist_dict = require_key(diffusion_dict, 'dist', 'model_params.diffusion')
    
    diffusion_config = DiffusionConfig(
        embedding_mask_proba=require_key(diffusion_dict, 'embedding_mask_proba', 'model_params.diffusion'),
        transformer=DiffusionTransformerConfig(
            num_layers=require_key(transformer_dict, 'num_layers', 'model_params.diffusion.transformer'),
            num_heads=require_key(transformer_dict, 'num_heads', 'model_params.diffusion.transformer'),
            head_features=require_key(transformer_dict, 'head_features', 'model_params.diffusion.transformer'),
            multiplier=require_key(transformer_dict, 'multiplier', 'model_params.diffusion.transformer')
        ),
        dist=DiffusionDistributionConfig(
            sigma_data=require_key(dist_dict, 'sigma_data', 'model_params.diffusion.dist'),
            estimate_sigma_data=require_key(dist_dict, 'estimate_sigma_data', 'model_params.diffusion.dist'),
            mean=require_key(dist_dict, 'mean', 'model_params.diffusion.dist'),
            std=require_key(dist_dict, 'std', 'model_params.diffusion.dist')
        )
    )
    
    slm_dict = require_key(model_dict, 'slm', 'model_params')
    slm_config = SLMConfig(
        model=require_key(slm_dict, 'model', 'model_params.slm'),
        sr=require_key(slm_dict, 'sr', 'model_params.slm'),
        hidden=require_key(slm_dict, 'hidden', 'model_params.slm'),
        nlayers=require_key(slm_dict, 'nlayers', 'model_params.slm'),
        initial_channel=require_key(slm_dict, 'initial_channel', 'model_params.slm')
    )
    
    converted['model_params'] = ModelConfig(
        multispeaker=require_key(model_dict, 'multispeaker', 'model_params'),
        max_style_length=require_key(style_tts2_config, 'max_len'), # original style tts 2 doesn't have this key
        dim_in=require_key(model_dict, 'dim_in', 'model_params'),
        hidden_dim=require_key(model_dict, 'hidden_dim', 'model_params'),
        max_conv_dim=require_key(model_dict, 'max_conv_dim', 'model_params'),
        n_layer=require_key(model_dict, 'n_layer', 'model_params'),
        n_mels=require_key(model_dict, 'n_mels', 'model_params'),
        n_token=require_key(model_dict, 'n_token', 'model_params'),
        max_dur=require_key(model_dict, 'max_dur', 'model_params'),
        style_dim=require_key(model_dict, 'style_dim', 'model_params'),
        dropout=require_key(model_dict, 'dropout', 'model_params'),
        plbert_config=model_dict.get('plbert_config', plbert_config),
        decoder=decoder_config,
        diffusion=diffusion_config,
        preprocess=converted['preprocess_params']
    )

    del converted['preprocess_params']
    
    converted['slm'] = slm_config
    
    loss_dict = require_key(style_tts2_config, 'loss_params')
    _default_loss = LossConfig()
    converted['loss_params'] = LossConfig(
        lambda_mel=require_key(loss_dict, 'lambda_mel', 'loss_params'),
        lambda_gen=require_key(loss_dict, 'lambda_gen', 'loss_params'),
        lambda_slm=require_key(loss_dict, 'lambda_slm', 'loss_params'),
        lambda_mono=require_key(loss_dict, 'lambda_mono', 'loss_params'),
        lambda_s2s=require_key(loss_dict, 'lambda_s2s', 'loss_params'),
        TMA_epoch=loss_dict.get('TMA_epoch', _default_loss.TMA_epoch),
        lambda_F0=require_key(loss_dict, 'lambda_F0', 'loss_params'),
        lambda_norm=require_key(loss_dict, 'lambda_norm', 'loss_params'),
        lambda_dur=require_key(loss_dict, 'lambda_dur', 'loss_params'),
        lambda_ce=require_key(loss_dict, 'lambda_ce', 'loss_params'),
        lambda_sty=require_key(loss_dict, 'lambda_sty', 'loss_params'),
        lambda_diff=require_key(loss_dict, 'lambda_diff', 'loss_params'),
        diff_epoch=require_key(loss_dict, 'diff_epoch', 'loss_params'),
        joint_epoch=require_key(loss_dict, 'joint_epoch', 'loss_params')
    )
    
    opt_dict = require_key(style_tts2_config, 'optimizer_params')
    _default_opt = OptimizerConfig()
    converted['optimizer_params'] = OptimizerConfig(
        lr=require_key(opt_dict, 'lr', 'optimizer_params'),
        max_lr=opt_dict.get('max_lr', _default_opt.max_lr),
        bert_lr=require_key(opt_dict, 'bert_lr', 'optimizer_params'),
        ft_lr=require_key(opt_dict, 'ft_lr', 'optimizer_params'),
        pct_start=opt_dict.get('pct_start', _default_opt.pct_start),
        div_factor=opt_dict.get('div_factor', _default_opt.div_factor),
        final_div_factor=opt_dict.get('final_div_factor', _default_opt.final_div_factor),
        max_grad_norm=opt_dict.get('max_grad_norm', _default_opt.max_grad_norm),
        weight_decay=opt_dict.get('weight_decay', _default_opt.weight_decay),
        betas=tuple(opt_dict.get('betas', _default_opt.betas))
    )
    
    slmadv_dict = require_key(style_tts2_config, 'slmadv_params')
    converted['slmadv_params'] = SLMAdvConfig(
        min_len=require_key(slmadv_dict, 'min_len', 'slmadv_params'),
        max_len=require_key(slmadv_dict, 'max_len', 'slmadv_params'),
        batch_percentage=require_key(slmadv_dict, 'batch_percentage', 'slmadv_params'),
        iter=require_key(slmadv_dict, 'iter', 'slmadv_params'),
        thresh=require_key(slmadv_dict, 'thresh', 'slmadv_params'),
        scale=require_key(slmadv_dict, 'scale', 'slmadv_params'),
        sig=require_key(slmadv_dict, 'sig', 'slmadv_params')
    )
    
    return TrainingArgs(**converted)

# @dataclass
# class TTSConfig:
    
#     first_stage_path: Optional[str] = None
#     save_freq: int = 2
#     audio_log_freq: int = 2
#     log_interval: int = 10
#     device: str = "cuda"
#     epochs_1st: int = 100
#     epochs_2nd: int = 50
#     batch_size: int = 16
#     max_len: int = 400
#     grad_checkpoint_gans: bool = False
#     grad_checkpoint_generator: bool = True
#     pretrained_model: Optional[str] = None
#     second_stage_load_pretrained: bool = True
#     load_only_params: bool = False

#     F0_path: str = "models/jdc/bst.t7"
#     ASR_config: str = "models/asr/config.yml"
#     ASR_path: str = "models/asr/epoch_00080.pth"
#     PLBERT_dir: str = "models/plbert_multilingual/"
    
#     data_params: DataConfig = field(default_factory=DataConfig)
#     preprocess_params: PreprocessConfig = field(default_factory=PreprocessConfig)
#     model_params: ModelConfig = field(default_factory=ModelConfig)
#     loss_params: LossConfig = field(default_factory=LossConfig)
#     optimizer_params: OptimizerConfig = field(default_factory=OptimizerConfig)
#     slmadv_params: SLMAdvConfig = field(default_factory=SLMAdvConfig)

class Stage(Enum):
    FIRST = "first"
    SECOND = "second"
    THIRD = "third"
    FINETUNE = "finetune"