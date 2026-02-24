import numpy as np
import random
import soundfile as sf

from utils import base_dir, model_id, voice_id, load_model

POLISH_SENTENCES = (
    "To jest pierwszy krótki test parametrów syntezy mowy. "
    "Jakość głosu zależy od ustawień beta, alpha oraz kroków dyfuzji. "
    "Czwarte zdanie służy do oceny brzmienia przy różnych wartościach. "
    "Piąta fraza pozwala usłyszeć wpływ współczynnika interpolacji stylu. "
    "Szóste zdanie w języku polskim sprawdza naturalność wypowiedzi. "
    "Ostatnie, siódme zdanie kończy ten zestaw testowy."
)

OUTPUT_DIR = base_dir / "output_wavs" / "grid_tts_params"


def save_audio(audio_chunk, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), audio_chunk.audio, audio_chunk.sample_rate)


def random_tts_params():
    return {
        "use_diffusion": True,
        "beta": round(random.uniform(0.05, 0.8), 3),
        "alpha": round(random.uniform(0.05, 0.8), 3),
        "style_interpolation_factor": round(random.uniform(0.0, 0.5), 3),
        "diffusion_steps": random.randint(5, 20),
        "embedding_scale": round(random.uniform(0.6, 1.4), 3),
    }


def run_use_diffusion_comparison(tts):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base_params = {
        "beta": 0.3,
        "alpha": 0.3,
        "style_interpolation_factor": 0.0,
        "diffusion_steps": 10,
        "embedding_scale": 1.0,
    }
    for use_diffusion in [True, False]:
        params = {**base_params, "use_diffusion": use_diffusion}
        tts_params = {"tts_params": params}
        audio_chunk = tts.generate_full(model_id, voice_id, POLISH_SENTENCES, tts_params)
        name = f"comparison_use_diffusion_{use_diffusion}.wav"
        save_audio(audio_chunk, OUTPUT_DIR / name)
        print(f"Saved {name}")


def run_random_search(tts, num_runs: int = 20):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for i in range(num_runs):
        params = random_tts_params()
        tts_params = {"tts_params": params}
        audio_chunk = tts.generate_full(model_id, voice_id, POLISH_SENTENCES, tts_params)
        parts = [f"{k}_{v}" for k, v in sorted(params.items())]
        name = f"run_{i:03d}_" + "_".join(parts)[:80] + ".wav"
        output_path = OUTPUT_DIR / name
        save_audio(audio_chunk, output_path)
        print(f"Saved {output_path.name}")


def main():
    tts = load_model()
    run_use_diffusion_comparison(tts)
    run_random_search(tts)
    tts.stop()
    print(f"Done. All files in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
