import sys, os, json, numpy as np, torch, soundfile as sf, logging, structlog
sys.path.insert(0, "/workspace/tinfer/tinfer")
structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.ERROR))
from tinfer.models.impl.styletts2.model.model import StyleTTS2
from tinfer.models.impl.styletts2.model.phonemizer import StyleTTS2Phonemizer
from tinfer.models.impl.styletts2.model.modules.decoder_blocks import AdaIN1d
from tinfer.core.request import AlignmentType

HERE = "/workspace/tinfer/scratchpad_freeze"
OUTW = "/workspace/tinfer/output_wavs"
os.makedirs(OUTW, exist_ok=True)
MODEL = "/workspace/converted_models/vokan/model.pth"
VOICE = "libri_f1"
TEXT = "The quick brown fox jumps over the lazy dog, and then it ran across the wide green field."

m = StyleTTS2(device="cuda")
m.load(MODEL, device="cuda", compile_model=False, load_style_encoder=True, runtime_engine="torch")
m._phonemizer = StyleTTS2Phonemizer(language="en-us")
m.load_voices_from_folder("/workspace/converted_models/vokan/voices")
decoder = m._model.decoder
adains = [mod for mod in decoder.modules() if isinstance(mod, AdaIN1d)]
print(f"AdaIN InstanceNorm sites: {len(adains)}")

def synth(tag, path):
    torch.manual_seed(1234)
    res = m.generate(TEXT, {"voice_id": VOICE},
        {"use_diffusion": False, "alpha": 0.3, "beta": 0.7, "embedding_scale": 1.0, "speed": 1.0, "diffusion_steps": 5},
        {"alignment_type": AlignmentType.NONE, "request_id": tag, "chunk_index": 0})
    a = np.asarray(res.data).astype(np.float32).squeeze()
    sf.write(path, a, res.sample_rate)
    print(f"{tag}: wrote {path} dur={len(a)/res.sample_rate:.2f}s peak={np.abs(a).max():.3f} rms={np.sqrt((a**2).mean()):.4f}")
    return a, res.sample_rate

# 1) BASELINE: real per-utterance InstanceNorm (full-sequence dependent)
base, sr = synth("baseline", f"{OUTW}/baseline_instancenorm.wav")

# 2) FROZEN: replace InstanceNorm with fixed per-channel (mu,sig) precomputed from data.
#    This normalization does NOT depend on the full sequence -> streaming/causal friendly.
z = np.load(f"{HERE}/frozen_stats.npz")
class FrozenNorm(torch.nn.Module):
    def __init__(self, mu, sig):
        super().__init__()
        self.register_buffer("mu",  torch.tensor(mu).view(1,-1,1))
        self.register_buffer("sig", torch.tensor(sig).view(1,-1,1))
    def forward(self, x):
        return (x - self.mu) / self.sig

for i,a in enumerate(adains):
    a.norm = FrozenNorm(z[f"mu_{i}"], z[f"sig_{i}"]).to("cuda")

froz, sr = synth("frozen", f"{OUTW}/frozen_normalization.wav")

# 3) numeric comparison (align lengths)
n = min(len(base), len(froz))
b, f = base[:n], froz[:n]
corr = float(np.corrcoef(b, f)[0,1])
rel_rms = float(np.sqrt(((b-f)**2).mean()) / (np.sqrt((b**2).mean()) + 1e-9))

# crude log-mel spectral distance (STFT magnitude, log) for perceptual proximity
def logmag(x):
    win = 1024; hop = 256
    xs = torch.tensor(x)
    S = torch.stft(xs, n_fft=win, hop_length=hop, window=torch.hann_window(win),
                   return_complex=True)
    return torch.log(S.abs() + 1e-5)
lb, lf = logmag(b), logmag(f)
k = min(lb.shape[1], lf.shape[1])
spec_l1 = float((lb[:, :k] - lf[:, :k]).abs().mean())

summary = {
  "text": TEXT, "voice": VOICE, "sr": sr,
  "dur_baseline_s": round(len(base)/sr, 3), "dur_frozen_s": round(len(froz)/sr, 3),
  "peak_baseline": round(float(np.abs(base).max()),4), "peak_frozen": round(float(np.abs(froz).max()),4),
  "rms_baseline": round(float(np.sqrt((base**2).mean())),5), "rms_frozen": round(float(np.sqrt((froz**2).mean())),5),
  "waveform_correlation": round(corr,4),
  "relative_rms_error": round(rel_rms,4),
  "logmel_l1_distance": round(spec_l1,4),
}
print("\n=== BASELINE vs FROZEN ===")
for kk,vv in summary.items(): print(f"  {kk}: {vv}")
json.dump(summary, open(f"{HERE}/compare_result.json","w"), indent=2)
print("\nsaved compare_result.json + wavs in output_wavs/")
