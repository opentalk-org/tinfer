from config import output_dir
from utils import save_audio, synthesize


def main() -> None:
    audio = synthesize("Dzień dobry. To jest przykład syntezy przez serwer Tinfer Rust.")
    save_audio(audio, output_dir / "basic.wav")


if __name__ == "__main__":
    main()
