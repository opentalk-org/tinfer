# Diffusion Comparison Histograms Design

## Goal

Extend the phoneme-duration benchmark with two phonemes-per-second histograms and a second, otherwise identical synthesis run that bypasses diffusion.

## Run Profiles

The existing single command remains:

```bash
./test_speed/run_benchmark.sh
```

It produces two profiles:

- `test_speed/results/`: diffusion enabled, preserving the current output path and behavior.
- `test_speed/results_no_diffusion/`: diffusion disabled with `use_diffusion=False` on every synthesis request.

Setting `alpha` or `beta` does not disable diffusion in this model. Those parameters only blend a sampled style with the reference style after diffusion has already run. The no-diffusion profile therefore uses the explicit `use_diffusion` inference parameter, which skips diffusion and conditions the predictor and decoder directly on the embedded reference style.

Both profiles use the same seed, 20 selected reference WAVs, four highlighted voices, 16 Polish texts, embeddings, and reporting calculations. The command extracts and embeds references once, copies the serialized references and embeddings into the no-diffusion folder, then reuses the loaded voice vectors for both synthesis profiles. Each result folder is therefore self-contained.

## Histograms

Each profile summary contains two global histogram PNG files with fixed, shared bin edges so enabled and disabled results can be compared directly:

- `phonemes_per_second_by_voice.png`: one value per voice, computed as that voice's arithmetic mean phonemes/s across its 16 text requests. The y-axis is the number of voices in each rate range and the total count is 20.
- `phonemes_per_second_all_runs.png`: one value per voice/text request. The y-axis is the number of runs in each rate range and the total count is 320.

Bin edges are computed once from the combined enabled and disabled values, using 0.25 phoneme/s-wide bins spanning the observed floor through ceiling. Both corresponding plots therefore use the same x-axis ranges and bins.

Existing scatter plots and phoneme-duration tables remain unchanged and are generated independently inside both profile folders.

## Components and Data Flow

The typed benchmark configuration gains a synthesis profile containing its name, result directory, and `use_diffusion` value. The inference layer accepts `use_diffusion` explicitly and includes it in every direct `StyleTTS2.generate()` parameter dictionary. The reporting layer accepts precomputed histogram edges and renders both histogram populations.

The benchmark disables `baseline_speed_corrected_for_request()` for both profiles. Because this is an experiment-specific requirement, no `tinfer/` source file or public inference parameter changes. Before synthesis, the runner monkeypatches the symbol imported by `tinfer.models.impl.styletts2.model.model` with an identity function that returns the requested speed unchanged. Both manifests record `speed_correction: false`.

The runner performs selection, model loading, and embedding once. It synthesizes the enabled profile, synthesizes the disabled profile, calculates shared histogram bins from both request sets, and writes each profile's raw metrics, summaries, plots, and manifest. Each manifest records the profile name and `use_diffusion` value, and each profile contains the same 20 reference WAVs and 20 serialized embeddings.

## Failure Behavior

The runner fails if either profile does not contain exactly 20 voices and 320 requests, if per-voice grouping does not yield 16 requests per voice, or if histogram counts do not sum to their expected population. Existing failures for missing inputs, internal text splitting, missing predictor alignments, and invalid durations remain unchanged.

## Verification

Unit tests verify that synthesis forwards both `use_diffusion=True` and `use_diffusion=False`, the speed-correction monkeypatch preserves the requested speed, per-voice averaging produces one value per voice, 0.25-wide shared bins cover both profiles, and both histogram files are emitted. The full command must produce 640 non-empty WAV files across the two folders, 20 embeddings in each profile, and two histograms plus the existing five scatters and five tables per profile.
