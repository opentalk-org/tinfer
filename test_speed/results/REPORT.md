# StyleTTS2 Duration Investigation

This is the authoritative index for the duration and speaking-rate experiments.
Generated artifacts live under `final/`; compact causal evidence lives under
`evidence/`. Failed and superseded audio runs were removed after their results
were recorded here.

## Final result layout

| Location | Contents |
|---|---|
| `final/magda_rate/nested` | Final Magda model, original 48 nested-prefix texts |
| `final/magda_rate/nested_no_diffusion` | Same predictor with diffusion disabled |
| `final/magda_rate/independent` | Final Magda model, 48 independent-span texts |
| `final/magda_rate/independent_no_diffusion` | Independent-span, diffusion disabled |
| `final/magda_original` | Original Magda baseline profiles |
| `final/agnieszka` | Agnieszka benchmark profiles |
| `final/olam` | Single-voice Olam benchmark profiles |
| `final/vokan` | Vokan with valid English inputs and LJSpeech reference |
| `final/ljspeech` | LJSpeech with valid English inputs and reference |
| `final/libri` | Libri benchmark using its three voice samples |
| `final/styletts_finetune_epoch10` | Backend epoch-10 finetune with backend references |

Every benchmark profile contains its manifest, generated audio, reference
audio, embeddings, request/phoneme CSVs, and summary plots.

## Final Magda measurements

The authoritative measurements use 20 voices and 48 texts without inference
chunking or speed correction.

| Corpus | Region | ph/s slope per input token |
|---|---:|---:|
| Nested-prefix | `<100` | `0.002994` |
| Nested-prefix | `>=100` | `0.000113` |
| Independent-span | `<100` | `-0.004494` |
| Independent-span | `>=100` | `0.000604` |

The acceptance threshold was `abs(slope) < 0.01` below 100 tokens while the
long-input slope remained near zero. Diffusion and no-diffusion profiles have
identical predictor-duration measurements.

Held-out aligned-duration log-MAE improved from `0.337304` to `0.330597`.
The predicted first-four/interior duration ratio is `0.9013`; the genuine
alignment target is `0.9627`.

## What was established

1. The effect occurs at batch size one and is not inference chunking.
2. Correct masks alone do not remove it.
3. Packed BiLSTM inference removes padding leakage but does not remove the
   genuine short-input rate defect.
4. PLBERT context is causal but explains only about 22% of the slope; embedding
   norm explains less than 1%.
5. Duration rounding contributes only a small part of the slope.
6. Most nested-prefix divergence is concentrated near the phrase onset, but a
   fixed onset-only correction cannot satisfy both rate and alignment gates.
7. The training duration target uses 25 ms frames. The first duration-only
   finetune incorrectly used 12.5 ms, causing a global timing shift.
8. The defect is style-conditioned. Selecting checkpoints on five voices gave
   misleading results; all 20 voices must be evaluated.

## Tested approaches

| Experiment | Outcome |
|---|---|
| Same tokens with 5 versus 50 right-padding tokens | Disproved padding as the production cause at batch size one |
| Packed `predictor.lstm` monkey patch | Correct padding behavior, but original short/long rate defect remained |
| BERT plus 100 context tokens, cropped before prediction | Modest improvement only; did not solve the slope |
| PLBERT-only finetuning | Rejected because PLBERT explained a minority of the effect |
| Incorrect 12.5 ms duration finetune | Failed; shifted global speaking rate |
| Four fixed/learned onset scales | Improved onset alignment but did not improve the rate slope |
| Small conditioned onset correction head | Improved onset log-MAE but did not improve the rate slope |
| Uniform onset scale `0.54` | Reduced short slope but materially damaged aligned error |
| Random synthetic-corpus rate regularization | Near-zero training control slope, poor transfer to real benchmark texts |
| Direct length-rate covariance loss | Worsened nested short slope to `0.02898` |
| Variance loss, weight 20, 400 steps | Full nested short slope `0.02552` |
| Variance loss, weight 100, 400 steps | Full nested short slope `0.01818` |
| Variance loss, weight 500, 400 steps | Continuous nested short slope `0.01579` |
| Variance loss, weight 500, 800 steps | Passed both full-pipeline corpora |

Compact evidence for the padding, BERT, duration-path, genuine-length, forced
alignment, and final-training investigations is retained in `evidence/`.

## Successful solution

The final checkpoint is:

```text
/workspace/converted_models/magda_aligned_rate/model.pth
```

PLBERT and the acoustic stack are frozen. `bert_encoder` and the conditioned
duration predictor are trained with two objectives:

```text
loss = genuine_aligned_duration_loss + 500 * utterance_rate_variance_loss
```

The aligned loss anchors every real token to a genuine forced-alignment target.
The rate loss uses length-balanced nested and independent corpora with randomly
sampled styles. Checkpoints are accepted only when held-out aligned error stays
below the original aligned baseline, then ranked using all four all-voice
short/long slopes.

This is deliberately not a runtime speed correction. Inference code and the
real phoneme sequence are unchanged.

## Normal-training proposal

The regularizer can be introduced during ordinary training after alignments
stabilize, with a gradual weight warm-up. The weight `500` is specific to the
short 800-step corrective run and must not be copied without validating loss
magnitudes over a full training schedule.

An unconditional utterance-rate variance penalty can suppress legitimate
accent, emphasis, and emotion. A production training integration should keep
aligned duration dominant and preferably regularize a residual conditioned on
phoneme identity, stress, punctuation, phrase position, and style, or compare
matched content under different context lengths. Validate stressed and
unstressed vowel durations separately.

## Reproduction

Run the final aligned training, rate training, and four-profile benchmark:

```bash
./test_speed/solution/run.sh
```

Run all general model benchmarks:

```bash
./test_speed/run.sh
```

The final pipeline uses tqdm progress, genuine Polish phonemization, 20 Magda
voices, 48 texts, predictor durations, no chunking, and no speed correction.
