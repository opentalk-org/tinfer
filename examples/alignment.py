from config import output_dir
from utils import save_audio, synthesize_timed


def main() -> None:
    result = synthesize_timed("Zażółć gęślą jaźń. To jest przykład synchronizacji tekstu.")
    save_audio(result.audio, output_dir / "alignment.wav")
    for character, start, end in zip(
        result.alignment.characters,
        result.alignment.character_start_times_seconds,
        result.alignment.character_end_times_seconds,
        strict=True,
    ):
        print(f"{character!r}: {start:.3f}s–{end:.3f}s")


if __name__ == "__main__":
    main()
