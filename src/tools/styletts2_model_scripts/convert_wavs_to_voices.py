import argparse
from pathlib import Path
import torch

from tinfer.models.impl.styletts2.model.modules.load_utils import load_model
from tinfer.models.impl.styletts2.voice.encoder import StyleTTS2VoiceEncoder

def convert_wavs_to_voices(wav_path: str, model_path: str, output_path: str | None = None):
    wav_path = Path(wav_path)
    if not wav_path.exists():
        raise ValueError(f"WAV file does not exist: {wav_path}")
    
    model_path = Path(model_path)
    if not model_path.exists():
        raise ValueError(f"Model file does not exist: {model_path}")
    
    if output_path is None:
        output_path = f"{wav_path.parent}/voices"
    else:
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
    
    wav_files = sorted(wav_path.glob("*.wav"))
    if len(wav_files) == 0:
        raise ValueError(f"No WAV files found in {wav_path}")
    

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Found {len(wav_files)} WAV files")
    print(f"Loading model from {model_path}")
    print(f"Using device: {device}")

    model, model_config = load_model(str(model_path), load_style_encoder=True)
    model = {key: model[key].to(device) for key in model}
    _ = [model[key].eval() for key in model]

    sample_rate = model_config.preprocess.sr

    voice_encoder = StyleTTS2VoiceEncoder(
        model=model,
        device=device,
        sample_rate=sample_rate
    )

    print("Processing WAV files...")
    for i, wav_file in enumerate(wav_files, 1):
        print(f"  [{i}/{len(wav_files)}] Processing {wav_file.name}")
        
        voice_vector = voice_encoder.compute_style_from_audio(str(wav_file))

        torch.save(voice_vector,f"{output_path}/{wav_file.stem}.pth")

    print("Conversion complete!")

def main():
    parser = argparse.ArgumentParser(
        description="Convert folder of WAV files to voice PTH embeddings"
    )

    parser.add_argument("--wav_dir", type=str, required=True, help="Path to folder containing WAV files")
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to model.pth file"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output folder (default: voices/)"
    )
    args = parser.parse_args()
    convert_wavs_to_voices(args.wav_dir, args.model_path, args.output)

if __name__ == "__main__":
    main()