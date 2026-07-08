import sys, os, json, numpy as np, torch, soundfile as sf, logging, structlog
sys.path.insert(0, "/workspace/tinfer/tinfer")
structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.ERROR))
from tinfer.models.impl.styletts2.model.model import StyleTTS2
from tinfer.models.impl.styletts2.model.phonemizer import StyleTTS2Phonemizer
from tinfer.models.impl.styletts2.model.modules.decoder_blocks import AdaIN1d
from tinfer.core.request import AlignmentType

HERE = "/workspace/tinfer/scratchpad_freeze"
OUTW = "/workspace/tinfer/output_wavs"
MODEL = "/workspace/converted_models/libri/model.pth"
VOICE = "libri_f1"
TEXT = ("The old lighthouse had stood on the rocky point for over a century, weathering "
        "storms that swallowed lesser buildings whole. Every evening its keeper climbed the "
        "spiral staircase, lit the great lamp, and watched the beam sweep across the restless "
        "water. Sailors far out at sea trusted that light more than any map or compass they "
        "carried. On the calmest nights, when the fog rolled in thick and silent, it was the "
        "only thing standing between them and the jagged rocks below.")

m = StyleTTS2(device="cuda")
m.load(MODEL, device="cuda", compile_model=False, load_style_encoder=True, runtime_engine="torch")
m._phonemizer = StyleTTS2Phonemizer(language="en-us")
m.load_voices_from_folder("/workspace/converted_models/libri/voices")
decoder = m._model.decoder
adains = [mod for mod in decoder.modules() if isinstance(mod, AdaIN1d)]
print(f"AdaIN InstanceNorm sites: {len(adains)} | words: {len(TEXT.split())}")

GEN = {"use_diffusion": False, "alpha": 0.3, "beta": 0.7, "embedding_scale": 1.0,
       "speed": 1.0, "diffusion_steps": 5}

def synth(tag, path):
    torch.manual_seed(1234)
    res = m.generate(TEXT, {"voice_id": VOICE}, GEN,
        {"alignment_type": AlignmentType.NONE, "request_id": tag, "chunk_index": 0})
    a = np.asarray(res.data).astype(np.float32).squeeze()
    sf.write(path, a, res.sample_rate)
    print(f"{tag}: {path} dur={len(a)/res.sample_rate:.2f}s peak={np.abs(a).max():.3f} rms={np.sqrt((a**2).mean()):.4f}")
    return a, res.sample_rate

base, sr = synth("baseline", f"{OUTW}/libri_long_baseline_instancenorm.wav")

z = np.load(f"{HERE}/frozen_stats_libri.npz")
class FrozenNorm(torch.nn.Module):
    def __init__(self, mu, sig):
        super().__init__()
        self.register_buffer("mu",  torch.tensor(mu).view(1,-1,1))
        self.register_buffer("sig", torch.tensor(sig).view(1,-1,1))
    def forward(self, x):
        return (x - self.mu) / self.sig
for i,a in enumerate(adains):
    a.norm = FrozenNorm(z[f"mu_{i}"], z[f"sig_{i}"]).to("cuda")
froz, sr = synth("frozen", f"{OUTW}/libri_long_frozen_normalization.wav")

n = min(len(base), len(froz)); b, f = base[:n], froz[:n]
corr = float(np.corrcoef(b, f)[0,1])
rel_rms = float(np.sqrt(((b-f)**2).mean()) / (np.sqrt((b**2).mean()) + 1e-9))
def logmag(x):
    S = torch.stft(torch.tensor(x), n_fft=1024, hop_length=256,
                   window=torch.hann_window(1024), return_complex=True)
    return torch.log(S.abs() + 1e-5)
lb, lf = logmag(b), logmag(f); k = min(lb.shape[1], lf.shape[1])
spec_l1 = float((lb[:, :k] - lf[:, :k]).abs().mean())
summary = {"model": "libri", "text": TEXT, "words": len(TEXT.split()), "voice": VOICE, "sr": sr,
  "dur_baseline_s": round(len(base)/sr,3), "dur_frozen_s": round(len(froz)/sr,3),
  "peak_baseline": round(float(np.abs(base).max()),4), "peak_frozen": round(float(np.abs(froz).max()),4),
  "rms_baseline": round(float(np.sqrt((base**2).mean())),5), "rms_frozen": round(float(np.sqrt((froz**2).mean())),5),
  "waveform_correlation": round(corr,4), "relative_rms_error": round(rel_rms,4),
  "logmel_l1_distance": round(spec_l1,4)}
json.dump(summary, open(f"{HERE}/compare_result_libri_long.json","w"), indent=2)
print("\n=== BASELINE vs FROZEN (libri, long) ===")
for kk,vv in summary.items():
    if kk != "text": print(f"  {kk}: {vv}")
