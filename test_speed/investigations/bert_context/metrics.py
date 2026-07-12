from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class RegionalMetric:
    region: str
    count: int
    slope: float
    correlation: float
    mean_phonemes_per_second: float
    slope_ci_low: float
    slope_ci_high: float


def regional_metrics(
    tokens: np.ndarray,
    rates: np.ndarray,
    seed: int = 20260711,
) -> list[RegionalMetric]:
    regions = (("<100", tokens < 100), (">=100", tokens >= 100))
    rng = np.random.default_rng(seed)
    output = []
    for name, selected in regions:
        x_values = tokens[selected].astype(float)
        y_values = rates[selected].astype(float)
        slope = float(np.polyfit(x_values, y_values, 1)[0])
        correlation = float(np.corrcoef(x_values, y_values)[0, 1])
        bootstrap = np.empty(2_000)
        for index in range(bootstrap.size):
            sampled = rng.integers(0, len(x_values), len(x_values))
            bootstrap[index] = np.polyfit(
                x_values[sampled], y_values[sampled], 1
            )[0]
        output.append(
            RegionalMetric(
                name,
                len(x_values),
                slope,
                correlation,
                float(y_values.mean()),
                float(np.quantile(bootstrap, 0.025)),
                float(np.quantile(bootstrap, 0.975)),
            )
        )
    return output


def metrics_as_dicts(metrics: list[RegionalMetric]) -> list[dict[str, object]]:
    return [asdict(metric) for metric in metrics]
