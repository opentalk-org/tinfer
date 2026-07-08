import sys, os, json, numpy as np, torch, soundfile as sf, logging, structlog
sys.path.insert(0, "/workspace/tinfer/tinfer")
structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.ERROR))
from tinfer.models.impl.styletts2.model.model import StyleTTS2
from tinfer.models.impl.styletts2.model.phonemizer import StyleTTS2Phonemizer
from tinfer.models.impl.styletts2.model.modules.decoder_blocks import AdaIN1d
from tinfer.core.request import AlignmentType

OUT = "/workspace/scratchpad_freeze"
MODEL = "/workspace/converted_models/vokan/model.pth"
VOICE = "libri_f1"

m = StyleTTS2(device="cuda")
m.load(MODEL, device="cuda", compile_model=False, load_style_encoder=True, runtime_engine="torch")
m._phonemizer = StyleTTS2Phonemizer(language="en-us")
m.load_voices_from_folder("/workspace/converted_models/vokan/voices")
decoder = m._model.decoder

# index every AdaIN InstanceNorm site
adains = [mod for mod in decoder.modules() if isinstance(mod, AdaIN1d)]
norms  = [a.norm for a in adains]
print(f"AdaIN InstanceNorm sites: {len(adains)}")

def gen_params():
    return ({"use_diffusion": False, "alpha": 0.3, "beta": 0.7, "embedding_scale": 1.0,
             "speed": 1.0, "diffusion_steps": 5},
            {"alignment_type": AlignmentType.NONE, "request_id": "x", "chunk_index": 0})

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

# ---- Phase A: collect per-utterance per-channel instance stats ----
per_utt = {i: [] for i in range(len(norms))}          # list of (mean_t[C], std_t[C])
pooled  = {i: None for i in range(len(norms))}         # [sum, sumsq, count]

cur = {}
def mk_hook(i):
    def pre(module, args):
        x = args[0].detach()
        # InstanceNorm1d stats: per (batch,channel) over time dim=2
        mt = x.mean(dim=2).mean(dim=0)                 # [C]
        vt = x.var(dim=2, unbiased=False).mean(dim=0)  # [C] (avg over batch)
        cur[i] = (mt.float().cpu().numpy(), np.sqrt(vt.float().cpu().numpy() + 1e-5))
        # pooled over all time & batch & utts
        s  = x.sum(dim=(0,2)); ss = (x*x).sum(dim=(0,2)); n = x.shape[0]*x.shape[2]
        acc = pooled[i]
        if acc is None: pooled[i] = [s.double().cpu().numpy(), ss.double().cpu().numpy(), float(n)]
        else:
            acc[0]+=s.double().cpu().numpy(); acc[1]+=ss.double().cpu().numpy(); acc[2]+=n
    return pre

handles = [nrm.register_forward_pre_hook(mk_hook(i)) for i,nrm in enumerate(norms)]

for k,txt in enumerate(SENTS):
    cur.clear()
    torch.manual_seed(1234)
    p1,p2 = gen_params()
    m.generate(txt, {"voice_id": VOICE}, p1, p2)
    for i in range(len(norms)):
        if i in cur: per_utt[i].append(cur[i])
    print(f"  utt {k+1}/{len(SENTS)} done")

for h in handles: h.remove()

# ---- frozen constants: pooled per-channel mean/std over ALL data ----
mu_frozen, sig_frozen = {}, {}
for i,acc in pooled.items():
    s,ss,n = acc
    mu = s/n
    var = np.maximum(ss/n - mu*mu, 0.0)
    mu_frozen[i]  = mu.astype(np.float32)
    sig_frozen[i] = np.sqrt(var + 1e-5).astype(np.float32)

np.savez(f"{OUT}/frozen_stats.npz",
         **{f"mu_{i}": mu_frozen[i] for i in range(len(norms))},
         **{f"sig_{i}": sig_frozen[i] for i in range(len(norms))})

# ---- Qualification: how far do per-utterance instance stats stray from frozen? ----
# If freezing were exact, every utterance's (mean_t ~ mu) and (std_t ~ sig).
ratios = []      # std_t / sig_frozen  (should be ~1)
centers = []     # (mean_t - mu)/sig_frozen (should be ~0)
for i in range(len(norms)):
    mu, sig = mu_frozen[i], sig_frozen[i]
    for (mt, st) in per_utt[i]:
        ratios.append(st/sig)
        centers.append((mt-mu)/sig)
ratios = np.concatenate(ratios); centers = np.concatenate(centers)

def pct(a,q): return float(np.percentile(a,q))
summary = {
  "n_sites": len(norms), "n_utts": len(SENTS),
  "std_ratio_p05": pct(ratios,5), "std_ratio_p50": pct(ratios,50), "std_ratio_p95": pct(ratios,95),
  "std_ratio_min": float(ratios.min()), "std_ratio_max": float(ratios.max()),
  "center_p05": pct(centers,5), "center_p50": pct(centers,50), "center_p95": pct(centers,95),
  "center_absmax": float(np.abs(centers).max()),
  "frac_ratio_outside_0.8_1.25": float(np.mean((ratios<0.8)|(ratios>1.25))),
  "frac_center_abs_gt_0.5": float(np.mean(np.abs(centers)>0.5)),
}
print("\n=== FREEZE QUALIFICATION (per-utterance instance stats vs frozen constants) ===")
for k,v in summary.items(): print(f"  {k}: {v:.4f}" if isinstance(v,float) else f"  {k}: {v}")
json.dump(summary, open(f"{OUT}/qualification.json","w"), indent=2)
print("\nsaved frozen_stats.npz + qualification.json")
