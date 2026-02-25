import asyncio
import grpc
import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
from pathlib import Path

from tinfer.server.grpc import styletts_pb2, styletts_pb2_grpc

base_dir = Path(__file__).resolve().parent.parent.parent.parent
output_dir = base_dir / "output_wavs"

SERVER_ADDRESS = "localhost:50052"
SAMPLE_RATE = 24000

packs = [
    ("agnieszka", "66a4ecf82e2a7ae68b14add9_7.97_4.27"),
    ("magda", "magda_001"),
    ("olam", "any"),
]

model_name, voice_id = packs[1]

text = (
    "Dziś jest 2024-04-18. W kalendarzu pojawiło się wydarzenie: \"Spotkanie z Łukaszēm o 13:45\". Czy potrafisz przeczytać znaki nietypowe, takie jak: ñ, ü, ç, æ, ß oraz symbole: ©, ™, €, £? W poniedziałek 01/05/2023 – święto państwowe. Użytkownik Piotr napisał: 'Zażółć gęślą jaźń! 🎉'. Sprawdź, czy TTS wypowie poprawnie daty, symbole i znaki diakrytyczne."
)


def bytes_to_audio(audio_bytes: bytes, sample_rate: int) -> np.ndarray:
    audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
    audio_float = audio_int16.astype(np.float32) / 32767.0
    return audio_float


def plot_spectrogram_with_word_alignments(audio, sample_rate, word_alignments, title, output_path):
    fig, axes = plt.subplots(2, 1, figsize=(14, 6))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    duration_seconds = len(audio) / sample_rate

    spectrogram = librosa.stft(audio, n_fft=2048, hop_length=512)
    magnitude = np.abs(spectrogram)
    magnitude_db = librosa.amplitude_to_db(magnitude, ref=np.max)

    axes[0].imshow(magnitude_db, aspect='auto', origin='lower', extent=[0, duration_seconds, 0, sample_rate/2], cmap='viridis')
    axes[0].set_ylabel('Frequency (Hz)', fontsize=11)
    axes[0].set_title('Speech Spectrogram', fontsize=12, fontweight='bold')
    axes[0].set_xlim(0, duration_seconds)

    axes[1].set_title('Word Alignments', fontsize=12, fontweight='bold')
    if word_alignments:
        y_pos = 0
        for al in word_alignments:
            start_s = al.start_ms / 1000.0
            end_s = al.end_ms / 1000.0
            axes[1].barh(y_pos, end_s - start_s, left=start_s, height=0.8, alpha=0.6, color='green')
            axes[1].text(start_s + (end_s - start_s) / 2, y_pos, al.word,
                        ha='center', va='center', fontsize=9, fontweight='bold')
            y_pos += 1
        axes[1].set_ylabel('Word Index', fontsize=11)
        axes[1].set_xlabel('Time (seconds)', fontsize=11)
        axes[1].set_xlim(0, duration_seconds)
        axes[1].set_ylim(-0.5, len(word_alignments) - 0.5)
    else:
        axes[1].text(0.5, 0.5, 'No alignments available',
                     ha='center', va='center', transform=axes[1].transAxes, fontsize=11)
        axes[1].set_xlabel('Time (seconds)', fontsize=11)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    plt.close()


async def run_grpc_alignment():
    print("=== gRPC Word Alignments ===")
    print(f"Using model: {model_name} with voice: {voice_id}")

    async with grpc.aio.insecure_channel(SERVER_ADDRESS) as channel:
        stub = styletts_pb2_grpc.StyleTTSServiceStub(channel)

        request = styletts_pb2.SynthesizeRequest(
            text=text,
            config=styletts_pb2.SynthesisConfig(
                model_id=model_name,
                voice_id=voice_id,
                sample_rate_hz=SAMPLE_RATE
            )
        )

        print("Sending request...")
        response = await stub.Synthesize(request)

        audio_array = bytes_to_audio(response.audio_data, SAMPLE_RATE)
        output_dir.mkdir(parents=True, exist_ok=True)
        audio_path = output_dir / f"grpc_alignment_{model_name}_{voice_id}.wav"
        sf.write(str(audio_path), audio_array, SAMPLE_RATE)
        print(f"Audio saved to {audio_path}")

        output_path = output_dir / f"grpc_alignment_plot_{model_name}_{voice_id}.png"
        plot_spectrogram_with_word_alignments(
            audio_array,
            SAMPLE_RATE,
            list(response.alignments),
            "gRPC Word Alignments",
            output_path
        )


async def main():
    print("=" * 70)
    print("gRPC Alignment Example")
    print("=" * 70)
    print(f"Connecting to server at {SERVER_ADDRESS}")
    print("=" * 70)

    try:
        await run_grpc_alignment()
        print("\nDone!")
    except grpc.RpcError as e:
        print(f"\nError connecting to server: {e.code()} - {e.details()}")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
