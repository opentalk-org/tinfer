from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Tuple
import yaml


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
        config_data = asdict(self)
        with open(yaml_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)


class Stage(Enum):
    FIRST = "first"
    SECOND = "second"
    THIRD = "third"
    FINETUNE = "finetune"
