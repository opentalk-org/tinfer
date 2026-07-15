# StyleTTS2 Conversation Conditioning

## Status

Future fine-tuning design. This does not apply to the current unmodified StyleTTS2 streaming implementation.

## Goal

Condition a generated conversational turn on fixed text context and the final one or two seconds of the preceding spoken turn without conflating the previous speaker's identity with the requested output voice.

## Fixed text positions

Every A invocation uses 512 positions with fixed semantic locations:

```text
position 0       BOS
positions 1–64   pre-context
position 65      <turn> or <continue>
positions 66–447 core
positions 448–511 post-context
```

The core advances by at most 382 tokens. Empty pre, core, or post positions remain masked; later regions never shift into unused positions. This keeps training, ONNX export, TensorRT profiles, and runtime placement identical.

`<turn>` marks the first window of a conversational response. `<continue>` marks later windows from the same response.

## Conversation audio

The input is the final two seconds of the preceding turn, resampled to 24 kHz:

```text
context_audio:        float [B, 1, 48000]
context_audio_length: int   [B]
```

Short audio is left-padded so the newest sample always occupies the final position. The encoder produces fixed right-aligned memory at 50 positions per second:

```text
audio_memory: float [B, 100, D]
audio_mask:   bool  [B, 100]
```

The runtime encodes conversation audio once at the beginning of a turn and caches the memory on the GPU. Every text window in that turn reuses it. Generated audio does not replace the conversation memory during the same turn.

## Model integration

### Text encoding

Selected BERT blocks receive cross-attention adapters:

```text
text = text_self_attention(text)
text = text + gate * cross_attention(text, audio_memory, audio_mask)
```

Adapters after every second or third BERT block provide deep conditioning without modifying every layer. Gates start near zero for stable fine-tuning.

### Duration and prosody

An attentive pool converts audio memory into a conversation summary. Duration and prosody predictors receive it alongside text and voice style:

```text
duration_input = text_encoding + voice_style + conversation_summary
prosody_input  = expanded_text + voice_style + conversation_summary
```

This gives previous-turn rhythm, energy, and cadence a direct path to duration, F0, and noise predictions.

### Style diffusion

Style diffusion also receives conversation memory or its pooled summary. Voice reference remains the speaker-identity input; conversation audio represents conversational state. They must remain separate so the output does not copy the preceding speaker's identity.

## Acoustic contract

Fine-tuned acoustic execution uses fixed locations:

```text
frames 0–63     pre-context
frames 64–111   emitted core
frames 112–127  post-context
```

Only the 48 core frames are returned. Empty boundary context remains masked or padded and never changes the positions of other regions.

## Turn lifecycle

```text
receive previous-turn audio
    → resample and retain its final two seconds
    → encode and cache conversation memory
    → run first text window with <turn>
    → run remaining windows with <continue>
    → preserve the same conversation memory for the entire generated turn
    → replace memory when the next external turn arrives
```

Within-turn waveform continuity remains the responsibility of acoustic pre-context, retained style state, harmonic phase, RNG position, and overlap handling.

## Training

Training samples contain preceding-turn audio and target-turn text/audio. Context audio ends at the real conversational boundary and never includes target speech.

Use:

- matched preceding-turn audio for normal samples;
- occasional masked context so generation remains possible without audio;
- occasional context from a different conversation to discourage unconditional use;
- duration, F0, energy, style, and waveform losses so audio context affects prosody rather than only text representations;
- consistency losses on overlapping text and acoustic core regions.

When preceding audio commonly belongs to another speaker, speaker identity and conversation conditioning must use separate encoders and conditioning paths.

## Runtime API direction

The public stream eventually needs an operation that sets conversation context before text generation:

```text
set conversation audio(turn_id, PCM, sample_rate)
append text
generate
```

Changing conversation audio while a generated turn is active is invalid. The context is replaced only at a new turn boundary.

## Non-goals for the current runtime

- No conversation audio input is added before a compatible model is fine-tuned.
- The current model does not receive fixed text slot markers.
- Current BERT positions are not changed to the fixed pre/core/post layout.
- Current acoustic execution may use overlap windows as an approximation, but it does not claim equivalence to this trained contract.
