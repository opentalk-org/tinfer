import argparse
from concurrent.futures import ThreadPoolExecutor

from config import output_dir
from utils import save_audio, synthesize


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize concurrently with two loaded Rust models")
    parser.add_argument("first_model")
    parser.add_argument("first_voice")
    parser.add_argument("second_model")
    parser.add_argument("second_voice")
    args = parser.parse_args()
    with ThreadPoolExecutor(max_workers=2) as workers:
        first = workers.submit(synthesize, "Pierwszy model mówi równocześnie.", args.first_model, args.first_voice)
        second = workers.submit(synthesize, "Drugi model mówi równocześnie.", args.second_model, args.second_voice)
        save_audio(first.result(), output_dir / "first_model.wav")
        save_audio(second.result(), output_dir / "second_model.wav")


if __name__ == "__main__":
    main()
