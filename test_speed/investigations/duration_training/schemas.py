from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    steps: int
    seed: int = 20260711
    actual_batch_size: int = 12
    synthetic_batch_size: int = 4
    predictor_learning_rate: float = 2e-5
    encoder_learning_rate: float = 1e-5
    consistency_weight: float = 0.75
    shape_weight: float = 0.75
    boundary_tokens: int = 4
    evaluation_interval: int = 100


@dataclass(frozen=True)
class TrainingRecord:
    step: int
    loss: float
    actual_total_loss: float
    synthetic_total_loss: float
    consistency_loss: float
    shape_loss: float
    rate_drift: float | None
    validation_error: float | None
