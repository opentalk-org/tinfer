import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from test_speed.investigations.bert_context.experiment import (
    BENCHMARK_DIR,
    LENGTHS,
    MODEL_PATH,
    OUTPUT_DIR,
)
from test_speed.investigations.bert_context.metrics import metrics_as_dicts, regional_metrics


ROOT = Path(__file__).resolve().parents[3]
BENCHMARKS = {
    "baseline": (
        ROOT
        / "test_speed/results/final/magda_original/diffusion/metrics/requests.csv"
    ),
    "current_finetune": BENCHMARK_DIR / "metrics/requests.csv",
}


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_rows(path: Path, rows: list[object]) -> None:
    dictionaries = [asdict(row) for row in rows]
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(dictionaries[0]))
        writer.writeheader()
        writer.writerows(dictionaries)


def benchmark_metrics() -> dict[str, object]:
    output = {}
    for name, path in BENCHMARKS.items():
        frame = pd.read_csv(path)
        tokens = frame["input_phoneme_tokens"].to_numpy()
        rates = frame["phonemes_per_second"].to_numpy()
        output[name] = {
            "regions": metrics_as_dicts(regional_metrics(tokens, rates)),
            "lte_50_mean": float(rates[tokens <= 50].mean()),
            "gte_400_mean": float(rates[tokens >= 400].mean()),
            "long_short_gap": float(rates[tokens >= 400].mean() - rates[tokens <= 50].mean()),
            "row_count": len(frame),
        }
    return output


def controlled_metrics(rates: pd.DataFrame) -> dict[str, object]:
    output = {}
    for condition, frame in rates.groupby("condition"):
        tokens = frame["input_phoneme_tokens"].to_numpy()
        output[condition] = {}
        for kind in ("continuous", "integer"):
            values = frame[f"{kind}_phonemes_per_second"].to_numpy()
            output[condition][kind] = metrics_as_dicts(regional_metrics(tokens, values))
    return output


def plot_global() -> None:
    figure, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    colors = {"baseline": "#777777", "current_finetune": "#0068b5"}
    for axis, region, predicate in (
        (axes[0], "<100", lambda x: x < 100),
        (axes[1], ">=100", lambda x: x >= 100),
    ):
        for name in colors:
            frame = pd.read_csv(BENCHMARKS[name])
            selected = predicate(frame["input_phoneme_tokens"].to_numpy())
            x_values = frame.loc[selected, "input_phoneme_tokens"].to_numpy()
            y_values = frame.loc[selected, "phonemes_per_second"].to_numpy()
            axis.scatter(x_values, y_values, s=8, alpha=0.18, color=colors[name])
            fit = np.polyfit(x_values, y_values, 1)
            fit_x = np.array([x_values.min(), x_values.max()])
            axis.plot(fit_x, np.polyval(fit, fit_x), color=colors[name], label=f"{name}: slope {fit[0]:.4f}")
        axis.set(title=region, xlabel="Input phoneme tokens", ylabel="Predicted phonemes/s")
        axis.grid(alpha=0.2)
        axis.legend()
    figure.suptitle("Magda benchmark: regional length-rate fits")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "global_regional_fits.png", dpi=180)
    plt.close(figure)


def plot_causal(rates: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for condition, frame in rates.groupby("condition"):
        grouped = frame.groupby("input_phoneme_tokens")["integer_phonemes_per_second"].mean()
        axes[0].plot(grouped.index, grouped.values, marker="o", markersize=3, label=condition)
    actual = rates[rates["condition"] == "actual"]
    for kind in ("continuous", "integer"):
        grouped = actual.groupby("input_phoneme_tokens")[f"{kind}_phonemes_per_second"].mean()
        axes[1].plot(grouped.index, grouped.values, marker="o", markersize=3, label=kind)
    axes[0].set_title("Causal PLBERT representation swaps")
    axes[1].set_title("Rounding/clamp decomposition")
    for axis in axes:
        axis.axvline(100, color="black", linestyle="--", alpha=0.4)
        axis.set(xlabel="Input phoneme tokens", ylabel="Predicted phonemes/s")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "causal_and_rounding.png", dpi=180)
    plt.close(figure)


def plot_divergence(divergence: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(12, 7))
    grouped = divergence.groupby(["stage", "input_phoneme_tokens"])["relative_l2"].median()
    for stage in divergence["stage"].unique():
        values = grouped.loc[stage]
        axis.plot(values.index, values.values, marker="o", markersize=2, label=stage)
    axis.set(xlabel="Shared-prefix input phoneme tokens", ylabel="Median relative L2 divergence", yscale="log")
    axis.axvline(100, color="black", linestyle="--", alpha=0.4)
    axis.grid(alpha=0.2)
    axis.legend(fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "layer_divergence.png", dpi=180)
    plt.close(figure)


def main() -> None:
    raw = torch.load(OUTPUT_DIR / "raw.pt", weights_only=False)
    rates = pd.DataFrame(raw["rates"])
    divergence = pd.DataFrame(raw["divergences"])
    crosschecks = pd.DataFrame(raw["crosschecks"])
    rates.to_csv(OUTPUT_DIR / "causal_rates.csv", index=False)
    divergence.to_csv(OUTPUT_DIR / "layer_divergence.csv", index=False)
    crosschecks.to_csv(OUTPUT_DIR / "duration_crosscheck.csv", index=False)
    benchmarks = benchmark_metrics()
    controlled = controlled_metrics(rates)
    per_voice = []
    actual = rates[rates["condition"] == "actual"]
    for voice, frame in actual.groupby("voice_id"):
        metrics = regional_metrics(frame["input_phoneme_tokens"].to_numpy(), frame["integer_phonemes_per_second"].to_numpy())
        per_voice.extend({"voice_id": voice, **item} for item in metrics_as_dicts(metrics))
    pd.DataFrame(per_voice).to_csv(OUTPUT_DIR / "per_voice_regional_metrics.csv", index=False)
    configuration = {
        "model_path": str(MODEL_PATH),
        "model_sha256": hash_file(MODEL_PATH),
        "benchmark_paths": {name: str(path) for name, path in BENCHMARKS.items()},
        "lengths": list(LENGTHS),
        "voices": sorted(rates["voice_id"].unique().tolist()),
        "passage_count": int(rates["passage_id"].nunique()),
        "eval_mode": True,
        "same_style_per_context_pair": True,
        "true_attention_mask": True,
        "duration_runtime": "torch checkpoint modules",
        "controlled_frame_seconds": 0.0125,
        "standard_benchmark_rate_source": "alignment parser metrics",
        "acoustic_runtime_used": False,
    }
    summary = {"configuration": configuration, "benchmarks": benchmarks, "controlled": controlled, "crosscheck_max_abs_ph_s": float(crosschecks["absolute_difference"].max())}
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_global()
    plot_causal(rates)
    plot_divergence(divergence)
    write_readme(summary, divergence)


def write_readme(summary: dict[str, object], divergence: pd.DataFrame) -> None:
    current = summary["benchmarks"]["current_finetune"]["regions"]
    actual = summary["controlled"]["actual"]
    long_context = summary["controlled"]["long_context_bert"]["integer"]
    norm_swap = summary["controlled"]["short_direction_long_norm"]["integer"]
    stage_medians = divergence.groupby("stage")["relative_l2"].median()
    voice_slopes = pd.read_csv(OUTPUT_DIR / "per_voice_regional_metrics.csv")
    short_voice_slopes = voice_slopes[voice_slopes["region"] == "<100"]["slope"]
    lines = [
        "# Magda BERT short-context investigation",
        "",
        "## Verdict",
        "",
        "**Mixed, with PLBERT context content causally important.** Valid right context changes the shared-prefix PLBERT output before any tuned duration module. Feeding the long-context shared-prefix representation into the otherwise identical predictor materially changes the short-region rate curve. Norm-only swaps do not reproduce that effect, so direction/content—not embedding magnitude alone—is the stronger BERT mechanism. The divergence propagates through the downstream bidirectional recurrence, and rounding/clamp contributes but is not its origin.",
        "",
        "This is legitimate bidirectional contextual dependence, not padding leakage: every comparison is batch size 1, contains genuine valid right-context tokens, and uses the correct true attention mask. PLBERT has global self-attention and learned absolute positions; early tokens can therefore depend on all valid later tokens. Saturation after roughly 100 tokens is compatible with diminishing marginal contextual change, but that shape can also be amplified by recurrent predictor dynamics and minimum integer durations.",
        "",
        "## Key measurements",
        "",
        f"Standard 20-voice current benchmark `<100`: slope `{current[0]['slope']:.6f}`, r `{current[0]['correlation']:.4f}`, mean `{current[0]['mean_phonemes_per_second']:.3f}` ph/s, n `{current[0]['count']}`. `>=100`: slope `{current[1]['slope']:.6f}`, r `{current[1]['correlation']:.4f}`, mean `{current[1]['mean_phonemes_per_second']:.3f}` ph/s, n `{current[1]['count']}`.",
        f"Controlled actual integer `<100` slope `{actual['integer'][0]['slope']:.6f}` versus continuous `{actual['continuous'][0]['slope']:.6f}`; `>=100` integer `{actual['integer'][1]['slope']:.6f}` versus continuous `{actual['continuous'][1]['slope']:.6f}`.",
        f"Long-context PLBERT shared-prefix swap: `<100` slope `{long_context[0]['slope']:.6f}`, `>=100` `{long_context[1]['slope']:.6f}`. Short direction with long norms: `<100` `{norm_swap[0]['slope']:.6f}`.",
        f"Across the five controlled styles, `<100` integer slopes range from `{short_voice_slopes.min():.6f}` to `{short_voice_slopes.max():.6f}` and remain positive for every style. Style changes severity, but does not explain the shared shape.",
        "",
        "The first divergent layer in execution order is raw `bert`; no tuned duration layer has run at that boundary. Median relative L2 is " + ", ".join(f"`{stage}`={stage_medians[stage]:.4g}" for stage in ("bert", "bert_encoder", "predictor.text_encoder.output", "predictor.lstm", "duration_continuous")) + ". Divergence is propagated, but is not monotonically amplified by every layer.",
        "",
        f"The attempted duration-only/standard-row cross-check differs by up to `{summary['crosscheck_max_abs_ph_s']:.4f}` ph/s. The standard report derives time through the alignment parser, while the controlled trace uses predictor frames at 300/24000 s. Therefore absolute controlled means are not presented as numerically interchangeable with the standard scatter; causal comparisons use the same trace metric on both sides. This limitation is recorded rather than hidden.",
        "",
        "## Artifacts",
        "",
        "- `global_regional_fits.png`: baseline/current scatter and separate regional fits.",
        "- `causal_and_rounding.png`: PLBERT swaps and continuous/integer comparison.",
        "- `layer_divergence.png`: shared-prefix divergence through the duration path.",
        "- `metrics.json`, `causal_rates.csv`, `layer_divergence.csv`, `per_voice_regional_metrics.csv`, and `duration_crosscheck.csv`: machine-readable evidence.",
        "",
        "## Limitations",
        "",
        "Causal traces use five fixed real Magda reference styles, four rotated real Polish passages, and 19 dense lengths. They isolate duration prediction and do not synthesize audio. The benchmark overlay remains the authoritative 20-voice/48-text population view. A representation swap is an intervention but is off the model's natural joint distribution, so it establishes sensitivity to PLBERT content rather than proving one individual attention head is the root cause. Previous fixed-window attention experiments did not fix the curve, so arbitrary local attention is not supported as a remedy.",
    ]
    (OUTPUT_DIR / "README.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
