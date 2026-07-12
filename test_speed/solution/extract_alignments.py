from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
import hashlib
import json

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm

from runner.nodes.training.styletts.finetune.training.modules.asr.models import ASRCNN
from runner.nodes.training.styletts.finetune.training.utils import (
    length_to_mask,
    mask_from_lens,
    maximum_path,
)
from runner.nodes.text.runtime.symbols import DEFAULT_STYLETTS_SYMBOLS, TextCleaner
from test_speed.investigations.duration_training.corpus import BACKEND_URL, DATASET_ID


ALIGNER_CHECKPOINT = Path(
    "/workspace/styletts_studio_v2/.cache/runflow/assets/checkpoints/"
    "a5c676b548c719b2d95ea511b9a5f3bcc3abca26446d4bfb34b41537dc463a2e/"
    "model_30.pth"
)
OUTPUT_DIR = Path("/workspace/tinfer/test_speed/results/evidence/alignments")
OUTPUT_PATH = OUTPUT_DIR / "aligned_durations.json"
TO_MEL = torchaudio.transforms.MelSpectrogram(
    n_fft=2048,
    win_length=1200,
    hop_length=300,
    n_mels=80,
)


@dataclass(frozen=True)
class AlignedSegment:
    segment_id: str
    audio_file_id: str
    tokens: list[int]
    phonemes: str
    durations: list[float]
    mel_frames: int
    path_score: float


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_aligner() -> ASRCNN:
    checkpoint = torch.load(
        ALIGNER_CHECKPOINT,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    state = {
        key.removeprefix("module."): value
        for key, value in checkpoint["net"]["text_aligner"].items()
    }
    aligner = ASRCNN(
        input_dim=80,
        hidden_dim=256,
        n_token=178,
        token_embedding_dim=512,
    )
    aligner.load_state_dict(state)
    return aligner.eval().to("cuda")


def fetch_dataset() -> list[tuple[str, dict[str, object]]]:
    endpoint = f"{BACKEND_URL}/audio-files?dataset={DATASET_ID}&limit=200&sort=name"
    with urlopen(endpoint) as response:
        files = json.load(response)["rows"]
    segments = []
    for row in files:
        with urlopen(f"{BACKEND_URL}/audio-files/{row['id']}") as response:
            detail = json.load(response)
        segments.extend((str(row["id"]), segment) for segment in detail["segment_preview"])
    return segments


def read_audio(audio_file_id: str) -> tuple[np.ndarray, int]:
    with urlopen(f"{BACKEND_URL}/audio-files/{audio_file_id}/content") as response:
        data = response.read()
    wave, sample_rate = sf.read(BytesIO(data))
    if wave.ndim == 2:
        wave = wave[:, 0]
    return np.asarray(wave, dtype=np.float32), int(sample_rate)


def segment_mel(
    wave: np.ndarray,
    sample_rate: int,
    start: float,
    end: float,
) -> torch.Tensor:
    selected = wave[int(start * sample_rate) : int(end * sample_rate)]
    if sample_rate != 24_000:
        selected = librosa.resample(selected, orig_sr=sample_rate, target_sr=24_000)
    padded = np.concatenate((np.zeros(5000), selected, np.zeros(5000)))
    mel = TO_MEL(torch.from_numpy(padded).float())
    normalized = (torch.log(1e-5 + mel) + 4) / 4
    return normalized[:, : normalized.shape[1] - normalized.shape[1] % 2]


@torch.inference_mode()
def align_segment(
    aligner: ASRCNN,
    tokens: list[int],
    mel: torch.Tensor,
) -> tuple[list[float], float]:
    texts = torch.tensor([tokens], dtype=torch.long, device="cuda")
    mels = mel.unsqueeze(0).to("cuda")
    mel_lengths = torch.tensor([mel.shape[1]], device="cuda")
    text_lengths = torch.tensor([len(tokens)], device="cuda")
    mask = length_to_mask(mel_lengths // (2**aligner.n_down)).to("cuda")
    _, _, attention = aligner(mels, mask, texts)
    attention = attention.transpose(-1, -2)[..., 1:].transpose(-1, -2)
    path_mask = mask_from_lens(
        attention,
        text_lengths,
        mel_lengths // (2**aligner.n_down),
    )
    path = maximum_path(attention, path_mask)
    durations = path.sum(axis=-1)[0, : len(tokens)]
    selected_scores = attention[path.bool()]
    return durations.cpu().tolist(), float(selected_scores.mean())


def main() -> None:
    aligner = load_aligner()
    cleaner = TextCleaner(DEFAULT_STYLETTS_SYMBOLS)
    segments = fetch_dataset()
    audio_cache = {}
    aligned = []
    for audio_file_id, segment in tqdm(segments, desc="Align real segments", unit="segment"):
        if audio_file_id not in audio_cache:
            audio_cache[audio_file_id] = read_audio(audio_file_id)
        wave, sample_rate = audio_cache[audio_file_id]
        phonemes = str(segment["phon"])
        tokens = [0, *cleaner(phonemes), 0]
        mel = segment_mel(
            wave,
            sample_rate,
            float(segment["start"]),
            float(segment["end"]),
        )
        durations, score = align_segment(aligner, tokens, mel)
        aligned.append(
            AlignedSegment(
                str(segment["id"]),
                audio_file_id,
                tokens,
                phonemes,
                durations,
                mel.shape[1],
                score,
            )
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "aligner_checkpoint": str(ALIGNER_CHECKPOINT),
        "aligner_sha256": sha256(ALIGNER_CHECKPOINT),
        "training_token_convention": "BOS + phonemes + EOS",
        "duration_target": "StyleTTS2 text_aligner maximum_path",
        "segments": [asdict(item) for item in aligned],
    }
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(f"Aligned segments: {len(aligned)}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
