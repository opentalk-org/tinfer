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
MODEL = "/workspace/converted_models/libri/model.pth"
VOICE = "libri_f1"
TEXT = "The quick brown fox jumps over the lazy dog, and then it ran across the wide green field."
SENTS = [
 "The quick brown fox jumps over the lazy dog.",
 "She sells seashells by the seashore every summer.",
 "In the beginning the universe was created and this made a lot of people angry.",
 "Please remember to buy milk, eggs, and a loaf of fresh bread.",
 "The stock market rallied sharply after the surprising jobs report.",
 "How much wood would a woodchuck chuck if it could chuck wood?",
 "Quietly, the old clock in the hallway struck midnight.",
 "Scientists discovered a new species of frog deep in the rainforest.",
 "I cannot believe how fast this year has already gone by.",
 "We will meet at the corner cafe around half past seven tonight.",
]

m = StyleTTS2(device="cuda")
m.load(MODEL, device="cuda", compile_model=False, load_style_encoder=True, runtime_engine="torch")
m._phonemizer = StyleTTS2Phonemizer(language="en-us")
m.load_voices_from_folder("/workspace/converted_models/libri/voices")
decoder = m._model.decoder
adains = [mod for mod in decoder.modules() if isinstance(mod, AdaIN1d)]
norms = [a.norm for a in adains]
print(f"AdaIN InstanceNorm sites: {len(adains)}")

GEN = ({"use_diffusion": False, "alpha": 0.3, "beta": 0.7, "embedding_scale": 1.0,
        "speed": 1.0, "diffusion_steps": 5},
       {"alignment_type": AlignmentType.NONE, "request_id": "x", "chunk_index": 0})

# ---- Phase A: estimate frozen per-channel stats from corpus (pooled over time+batch+utts) ----
per_utt = {i: [] for i in range(len(norms))}
pooled = {i: None for i in range(len(norms))}
cur = {}
def mk_hook(i):
    def pre(module, args):
        x = args[0].detach()
        mt = x.mean(dim=2).mean(dim=0)
        vt = x.var(dim=2, unbiased=False).mean(dim=0)
        cur[i] = (mt.float().cpu().numpy(), np.sqrt(vt.float().cpu().numpy() + 1e-5))
        s = x.sum(dim=(0,2)); ss = (x*x).sum(dim=(0,2)); n = x.shape[0]*x.shape[2]
        acc = pooled[i]
        if acc is None: pooled[i] = [s.double().cpu().numpy(), ss.double().cpu().numpy(), float(n)]
        else:
            acc[0]+=s.double().cpu().numpy(); acc[1]+=ss.double().cpu().numpy(); acc[2]+=n
    return pre
handles = [nrm.register_forward_pre_hook(mk_hook(i)) for i,nrm in enumerate(norms)]
for k,txt in enumerate(SENTS):
    cur.clear(); torch.manual_seed(1234)
    m.generate(txt, {"voice_id": VOICE}, GEN[0], GEN[1])
    for i in range(len(norms)):
        if i in cur: per_utt[i].append(cur[i])
    print(f"  stats utt {k+1}/{len(SENTS)}")
for h in handles: h.remove()

mu_frozen, sig_frozen = {}, {}
for i,acc in pooled.items():
    s,ss,n = acc; mu = s/n
    var = np.maximum(ss/n - mu*mu, 0.0)
    mu_frozen[i] = mu.astype(np.float32)
    sig_frozen[i] = np.sqrt(var + 1e-5).astype(np.float32)
np.savez(f"{HERE}/frozen_stats_libri.npz",
         **{f"mu_{i}": mu_frozen[i] for i in range(len(norms))},
         **{f"sig_{i}": sig_frozen[i] for i in range(len(norms))})

ratios, centers = [], []
for i in range(len(norms)):
    mu, sig = mu_frozen[i], sig_frozen[i]
    for (mt, st) in per_utt[i]:
        ratios.append(st/sig); centers.append((mt-mu)/sig)
ratios = np.concatenate(ratios); centers = np.concatenate(centers)
def pct(a,q): return float(np.percentile(a,q))
qual = {"n_sites": len(norms), "n_utts": len(SENTS),
  "std_ratio_p05": pct(ratios,5), "std_ratio_p50": pct(ratios,50), "std_ratio_p95": pct(ratios,95),
  "std_ratio_min": float(ratios.min()), "std_ratio_max": float(ratios.max()),
  "center_p05": pct(centers,5), "center_p50": pct(centers,50), "center_p95": pct(centers,95),
  "center_absmax": float(np.abs(centers).max()),
  "frac_ratio_outside_0.8_1.25": float(np.mean((ratios<0.8)|(ratios>1.25))),
  "frac_center_abs_gt_0.5": float(np.mean(np.abs(centers)>0.5))}
json.dump(qual, open(f"{HERE}/qualification_libri.json","w"), indent=2)
print("qualification:", {k:round(v,3) if isinstance(v,float) else v for k,v in qual.items()})

# ---- Phase B: baseline vs frozen ----
def synth(tag, path):
    torch.manual_seed(1234)
    res = m.generate(TEXT, {"voice_id": VOICE}, GEN[0],
        {"alignment_type": AlignmentType.NONE, "request_id": tag, "chunk_index": 0})
    a = np.asarray(res.data).astype(np.float32).squeeze()
    sf.write(path, a, res.sample_rate)
    print(f"{tag}: {path} dur={len(a)/res.sample_rate:.2f}s peak={np.abs(a).max():.3f} rms={np.sqrt((a**2).mean()):.4f}")
    return a, res.sample_rate

base, sr = synth("baseline", f"{OUTW}/libri_baseline_instancenorm.wav")

class FrozenNorm(torch.nn.Module):
    def __init__(self, mu, sig):
        super().__init__()
        self.register_buffer("mu",  torch.tensor(mu).view(1,-1,1))
        self.register_buffer("sig", torch.tensor(sig).view(1,-1,1))
    def forward(self, x):
        return (x - self.mu) / self.sig
for i,a in enumerate(adains):
    a.norm = FrozenNorm(mu_frozen[i], sig_frozen[i]).to("cuda")
froz, sr = synth("frozen", f"{OUTW}/libri_frozen_normalization.wav")

n = min(len(base), len(froz)); b, f = base[:n], froz[:n]
corr = float(np.corrcoef(b, f)[0,1])
rel_rms = float(np.sqrt(((b-f)**2).mean()) / (np.sqrt((b**2).mean()) + 1e-9))
def logmag(x):
    S = torch.stft(torch.tensor(x), n_fft=1024, hop_length=256,
                   window=torch.hann_window(1024), return_complex=True)
    return torch.log(S.abs() + 1e-5)
lb, lf = logmag(b), logmag(f); k = min(lb.shape[1], lf.shape[1])
spec_l1 = float((lb[:, :k] - lf[:, :k]).abs().mean())
summary = {"model": "libri", "text": TEXT, "voice": VOICE, "sr": sr,
  "dur_baseline_s": round(len(base)/sr,3), "dur_frozen_s": round(len(froz)/sr,3),
  "peak_baseline": round(float(np.abs(base).max()),4), "peak_frozen": round(float(np.abs(froz).max()),4),
  "rms_baseline": round(float(np.sqrt((base**2).mean())),5), "rms_frozen": round(float(np.sqrt((froz**2).mean())),5),
  "waveform_correlation": round(corr,4), "relative_rms_error": round(rel_rms,4),
  "logmel_l1_distance": round(spec_l1,4)}
json.dump(summary, open(f"{HERE}/compare_result_libri.json","w"), indent=2)
print("\n=== BASELINE vs FROZEN (libri) ===")
for kk,vv in summary.items(): print(f"  {kk}: {vv}")
