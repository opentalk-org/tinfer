import sys, os, numpy as np, torch, soundfile as sf, logging, structlog
sys.path.insert(0, "/workspace/tinfer/tinfer")
structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.ERROR))
from tinfer.models.impl.styletts2.model.model import StyleTTS2
from tinfer.models.impl.styletts2.model.phonemizer import StyleTTS2Phonemizer
from tinfer.models.impl.styletts2.model.modules.decoder_blocks import AdaIN1d
from tinfer.core.request import AlignmentType

OUT = "/workspace/scratchpad_freeze"
MODEL = "/workspace/converted_models/vokan/model.pth"
VOICE = "libri_f1"
TEXT = "The quick brown fox jumps over the lazy dog."

m = StyleTTS2(device="cuda")
m.load(MODEL, device="cuda", compile_model=False, load_style_encoder=True, runtime_engine="torch")
m._phonemizer = StyleTTS2Phonemizer(language="en-us")
m.load_voices_from_folder("/workspace/converted_models/vokan/voices")
decoder = m._model.decoder
adains = [mod for mod in decoder.modules() if isinstance(mod, AdaIN1d)]

def synth(tag, path):
    torch.manual_seed(1234)
    res = m.generate(TEXT, {"voice_id": VOICE},
        {"use_diffusion": False, "alpha": 0.3, "beta": 0.7, "embedding_scale": 1.0, "speed": 1.0, "diffusion_steps": 5},
        {"alignment_type": AlignmentType.NONE, "request_id": tag, "chunk_index": 0})
    a = np.asarray(res.data).astype(np.float32).squeeze()
    sf.write(path, a, res.sample_rate)
    print(f"{tag}: wrote {path} dur={len(a)/res.sample_rate:.2f}s peak={np.abs(a).max():.3f} rms={np.sqrt((a**2).mean()):.4f}")
    return a, res.sample_rate

# 1) BASELINE (real per-utterance InstanceNorm)
base, sr = synth("baseline", f"/workspace/baseline_instancenorm.wav")

# 2) FROZEN: replace InstanceNorm with fixed per-channel (mu,sig) from data
z = np.load(f"{OUT}/frozen_stats.npz")
class FrozenNorm(torch.nn.Module):
    def __init__(self, mu, sig):
        super().__init__()
        self.register_buffer("mu",  torch.tensor(mu).view(1,-1,1))
        self.register_buffer("sig", torch.tensor(sig).view(1,-1,1))
    def forward(self, x):
        return (x - self.mu) / self.sig

for i,a in enumerate(adains):
    a.norm = FrozenNorm(z[f"mu_{i}"], z[f"sig_{i}"]).to("cuda")

froz, sr = synth("frozen", f"/workspace/frozen_instancenorm.wav")

# 3) numeric comparison (align lengths)
n = min(len(base), len(froz))
b, f = base[:n], froz[:n]
corr = float(np.corrcoef(b, f)[0,1])
rel_rms = float(np.sqrt(((b-f)**2).mean()) / (np.sqrt((b**2).mean()) + 1e-9))
print(f"\nBASELINE vs FROZEN:  len_base={len(base)} len_frozen={len(froz)}")
print(f"  waveform correlation = {corr:.4f}")
print(f"  relative RMS error   = {rel_rms:.4f}  ({rel_rms*100:.1f}% of signal RMS)")
