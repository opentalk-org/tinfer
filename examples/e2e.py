import sys, time
sys.path.insert(0, "/workspace/tinfer/tinfer")
import torch, torch.nn as nn
from tinfer.models.impl.styletts2.model.modules.load_utils import load_model

MODEL = "/workspace/converted_models/libri/model.pth"
dev = torch.device("cuda")
MAXF = 512      # fixed-max mel frames (alignment padded to this)
MAX_STEPS = 10  # diffusion steps runtime-selectable in [2, MAX_STEPS]

m, model_config = load_model(MODEL, load_style_encoder=True)
for k in m: m[k] = m[k].to(dev).eval()
for k in m:
    for p in m[k].parameters(): p.requires_grad_(False)
    for b in m[k].buffers(): b.requires_grad_(False)
# ---- export-friendly monkeypatches (plain LSTM, manual instance-norm) ----
import torch.nn.functional as F
from tinfer.models.impl.styletts2.model.modules import blocks as BLK

def te_forward(self, x, input_lengths, mm):
    x = self.embedding(x).transpose(1, 2)
    m2 = mm.unsqueeze(1)
    x = x.masked_fill(m2, 0.0)
    for c in self.cnn:
        x = c(x); x = x.masked_fill(m2, 0.0)
    x = x.transpose(1, 2)
    self.lstm.flatten_parameters()
    x, _ = self.lstm(x)
    x = x.transpose(-1, -2)
    return x.masked_fill(m2, 0.0)
BLK.TextEncoder.forward = te_forward

def de_forward(self, x, style, text_lengths, mm):
    masks = mm
    x = x.permute(2, 0, 1)
    s = style.expand(x.shape[0], x.shape[1], -1)
    x = torch.cat([x, s], dim=-1)
    x = x.masked_fill(masks.unsqueeze(-1).transpose(0, 1), 0.0)
    x = x.transpose(0, 1).transpose(-1, -2)
    for block in self.lstms:
        if isinstance(block, BLK.AdaLayerNorm):
            x = block(x.transpose(-1, -2), style).transpose(-1, -2)
            x = torch.cat([x, s.permute(1, 2, 0)], dim=1)
            x = x.masked_fill(masks.unsqueeze(-1).transpose(-1, -2), 0.0)
        else:
            x = x.transpose(-1, -2)
            block.flatten_parameters()
            x, _ = block(x)
            x = x.transpose(-1, -2)
    return x.transpose(-1, -2)
BLK.DurationEncoder.forward = de_forward

def adain_forward(self, x, s):
    h = self.fc(s).unsqueeze(-1)
    gamma, beta = torch.chunk(h, 2, dim=1)
    mu = x.mean(dim=2, keepdim=True)
    var = x.var(dim=2, keepdim=True, unbiased=False)
    xn = (x - mu) / torch.sqrt(var + 1e-5)
    return (1 + gamma) * xn + beta
BLK.AdaIN1d.forward = adain_forward

sigma_data = float(m.diffusion.diffusion.sigma_data)
net = m.diffusion.diffusion.net           # StyleTransformer1d
SMIN, SMAX, RHO = 1e-4, 3.0, 9.0          # KarrasSchedule(sigma_min,sigma_max,rho) from build
ADPM2_RHO = 1.0

def length_to_mask(lengths, L):
    idx = torch.arange(L, device=lengths.device).unsqueeze(0).expand(lengths.shape[0], -1)
    return torch.gt(idx + 1, lengths.unsqueeze(1))     # True = padding

def scale_weights(sigma):  # sigma: (B,) tensor
    s = sigma.view(-1,1,1)
    c_skip = sigma_data**2 / (s**2 + sigma_data**2)
    c_out  = s * sigma_data * (sigma_data**2 + s**2) ** -0.5
    c_in   = (s**2 + sigma_data**2) ** -0.5
    c_noise = torch.log(sigma) * 0.25
    return c_skip, c_out, c_in, c_noise

def denoise(x, sigma, emb, feat):
    c_skip, c_out, c_in, c_noise = scale_weights(sigma)
    x_pred = net(c_in * x, c_noise, embedding=emb, features=feat)
    return c_skip * x + c_out * x_pred

def karras_sigma(i, num_steps):   # closed form, i and num_steps are floats/tensors
    a = SMAX ** (1.0/RHO)
    b = SMIN ** (1.0/RHO)
    frac = i / (num_steps - 1)
    return (a + frac * (b - a)) ** RHO

def adpm2_step(x, emb, feat, sigma, sigma_next, step_noise):
    r = ADPM2_RHO
    sigma_up = torch.sqrt(torch.clamp(sigma_next**2 * (sigma**2 - sigma_next**2) / sigma**2, min=0))
    sigma_down = torch.sqrt(torch.clamp(sigma_next**2 - sigma_up**2, min=0))
    sigma_mid = ((sigma**(1/r) + sigma_down**(1/r)) / 2) ** r
    d = (x - denoise(x, sigma.expand(x.shape[0]), emb, feat)) / sigma
    x_mid = x + d * (sigma_mid - sigma)
    d_mid = (x_mid - denoise(x_mid, sigma_mid.expand(x.shape[0]), emb, feat)) / sigma_mid
    x = x + d_mid * (sigma_down - sigma)
    return x + step_noise * sigma_up

def sample_diffusion(noise, emb, feat, step_noise, ns):
    # ns: scalar float tensor (runtime num_steps). Masked unroll to MAX_STEPS.
    dev_ = noise.device
    sig0 = karras_sigma(torch.zeros((), device=dev_), ns)
    x = sig0 * noise
    for i in range(MAX_STEPS - 1):
        fi = torch.full((), float(i), device=dev_)
        si = karras_sigma(torch.clamp(fi,     max=ns - 1), ns)
        sj = karras_sigma(torch.clamp(fi + 1, max=ns - 1), ns)
        x_new = adpm2_step(x, emb, feat, si.view(1), sj.view(1), step_noise[:, i:i+1])
        active = (fi < ns - 1).float()      # 1.0 while i < num_steps-1, else 0.0
        x = active * x_new + (1.0 - active) * x
    return x

def shift1(t):   # hifigan right-shift along last dim
    return torch.cat([t[:, :, :1], t[:, :, :-1]], dim=-1)

def build_alignment(pred_dur, maxf):
    cum = torch.cumsum(pred_dur, dim=1)          # (B,L)
    start = cum - pred_dur
    fidx = torch.arange(maxf, device=pred_dur.device).view(1,1,-1)
    mask = (fidx >= start.unsqueeze(-1)) & (fidx < cum.unsqueeze(-1))
    return mask.float()                          # (B,L,maxf)

class E2E(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_encoder = m.text_encoder
        self.bert = m.bert
        self.bert_encoder = m.bert_encoder
        self.predictor = m.predictor
        self.decoder = m.decoder
    def forward(self, tokens, input_lengths, ref_s, diff_noise, step_noise, num_steps, alpha, beta):
        L = tokens.shape[1]
        tm = length_to_mask(input_lengths, L)
        t_en = self.text_encoder(tokens, input_lengths, tm)
        bert_dur = self.bert(tokens, attention_mask=(~tm).int())
        d_en = self.bert_encoder(bert_dur).transpose(-1, -2)
        s_pred = sample_diffusion(diff_noise, bert_dur, ref_s, step_noise, num_steps)[:, 0]
        s = s_pred[:, 128:]; ref = s_pred[:, :128]
        ref = alpha * ref + (1 - alpha) * ref_s[:, :128]
        s = beta * s + (1 - beta) * ref_s[:, 128:]
        d = self.predictor.text_encoder(d_en, s, input_lengths, tm)
        x, _ = self.predictor.lstm(d)
        duration = torch.sigmoid(self.predictor.duration_proj(x)).sum(-1)
        pred_dur = torch.round(duration).clamp(min=1) * (~tm).float()
        aln = build_alignment(pred_dur, MAXF)
        en = shift1(torch.bmm(d.transpose(-1, -2), aln))
        F0_pred, N_pred = self.predictor.F0Ntrain(en, s)
        asr = shift1(torch.bmm(t_en, aln))
        audio = self.decoder(asr, F0_pred, N_pred, ref)
        valid = pred_dur.sum(-1)
        return audio, valid

model = E2E().eval()
B, L = 1, 171
num_steps_t = torch.tensor(5.0, device=dev)
tokens = torch.randint(1, 178, (B, L), device=dev)
input_lengths = torch.full((B,), L, device=dev, dtype=torch.long)
ref_s = torch.randn(B, 256, device=dev)
diff_noise = torch.randn(B, 1, 256, device=dev)
step_noise = torch.randn(B, MAX_STEPS-1, 256, device=dev)
with torch.no_grad():
    for _ in range(2):
        audio, valid = model(tokens, input_lengths, ref_s, diff_noise, step_noise, num_steps_t, 0.7, 0.3)
    torch.cuda.synchronize()
    t0=time.time()
    for _ in range(10):
        audio, valid = model(tokens, input_lengths, ref_s, diff_noise, step_noise, num_steps_t, 0.7, 0.3)
    torch.cuda.synchronize(); dt=(time.time()-t0)/10*1000
print(f"OK audio {tuple(audio.shape)} valid {valid.tolist()} finite={torch.isfinite(audio).all().item()} "
      f"rng[{audio.min().item():.3f},{audio.max().item():.3f}] torch_time {dt:.1f}ms")

if "--export" in sys.argv:
    alpha_t = torch.tensor(0.7, device=dev); beta_t = torch.tensor(0.3, device=dev)
    # trace with batch=2 so batch-dependent broadcasts stay dynamic
    B2 = 2
    tokens2 = torch.randint(1,178,(B2,L),device=dev)
    il2 = torch.full((B2,), L, device=dev, dtype=torch.long)
    ref2 = torch.randn(B2,256,device=dev)
    dn2 = torch.randn(B2,1,256,device=dev)
    sn2 = torch.randn(B2,MAX_STEPS-1,256,device=dev)
    args = (tokens2, il2, ref2, dn2, sn2, num_steps_t, alpha_t, beta_t)
    dyn = {"tokens":{0:"B",1:"L"}, "input_lengths":{0:"B"}, "ref_s":{0:"B"},
           "diff_noise":{0:"B"}, "step_noise":{0:"B"}, "audio":{0:"B",2:"T"}, "valid":{0:"B"}}
    path = "/tmp/e2e.onnx"
    print("exporting ONNX ...", flush=True)
    torch.onnx.export(model, args, path, opset_version=17,
        input_names=["tokens","input_lengths","ref_s","diff_noise","step_noise","num_steps","alpha","beta"],
        output_names=["audio","valid"], dynamic_axes=dyn, do_constant_folding=True, dynamo=False)
    print("exported", path)
    import onnx
    o = onnx.load(path)
    print("nodes", len(o.graph.node), "inputs", [i.name for i in o.graph.input], "initializers", len(o.graph.initializer))
