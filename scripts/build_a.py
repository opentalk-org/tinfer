"""Build engine A = text_encoder + PLBERT + bert_encoder + diffusion(ADPM2) +
style blend + duration predictor (DurationEncoder + lstm + duration_proj).
Weights as inputs. Outputs: durations, d, t_en, s, ref for glue1/B/C.

LSTMs are reimplemented without pack_padded_sequence (plain LSTM), which is exact
for same-length batches (the benchmark case; production should length-bucket)."""
from __future__ import annotations
import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from build_common import load_model, strip_weight_norm, export_promote_build, TEXT_150
from tinfer.models.impl.styletts2.model.modules.blocks import AdaLayerNorm
from tinfer.models.impl.styletts2.model.modules.tensorrt_export import DiffusionTRTExportModule
from tinfer.models.impl.styletts2.model.trt3.core import WeightInputRunner, dump_tensors


def fast_lstm(lstm, x):
    """Bidirectional LSTM with the input projection done as ONE batched GEMM over all
    timesteps (TRT's nn.LSTM does it per-timestep -> thousands of tiny GEMMs at batch1).
    Only the recurrence loops. Matches torch.nn.LSTM (gate order i,f,g,o).

    TINFER_NATIVE_LSTM=1 skips the Python unroll and calls the nn.LSTM module directly,
    so ONNX exports one LSTM op instead of ~174 unrolled steps. That cuts the engine-A
    graph from ~65k nodes to a few thousand (build minutes -> seconds), at the cost of
    the static CUDA-graph-capturable unroll (TRT lowers LSTM to a dynamic loop)."""
    if os.environ.get("TINFER_NATIVE_LSTM"):
        # explicit batch-dependent h0/c0 so ONNX doesn't bake batch=1 into the initial
        # states (which breaks dynamic-batch profiles). x is [B,T,C] (batch_first).
        nd = (2 if lstm.bidirectional else 1) * lstm.num_layers
        h0 = x.new_zeros(nd, x.shape[0], lstm.hidden_size)
        c0 = x.new_zeros(nd, x.shape[0], lstm.hidden_size)
        return lstm(x, (h0, c0))[0]
    H = lstm.hidden_size
    B, T, _ = x.shape

    def run(Wih, Whh, bih, bhh, reverse):
        xp = F.linear(x, Wih, bih)  # [B,T,4H] batched input projection
        h = torch.zeros(B, H, device=x.device, dtype=x.dtype)
        c = torch.zeros(B, H, device=x.device, dtype=x.dtype)
        outs = []
        for s in range(T):
            t = T - 1 - s if reverse else s
            g = xp[:, t, :] + F.linear(h, Whh, bhh)
            i = torch.sigmoid(g[:, :H]); f = torch.sigmoid(g[:, H:2 * H])
            gg = torch.tanh(g[:, 2 * H:3 * H]); o = torch.sigmoid(g[:, 3 * H:])
            c = f * c + i * gg
            h = o * torch.tanh(c)
            outs.append(h)
        if reverse:
            outs = outs[::-1]
        return torch.stack(outs, dim=1)

    yf = run(lstm.weight_ih_l0, lstm.weight_hh_l0, lstm.bias_ih_l0, lstm.bias_hh_l0, False)
    yr = run(lstm.weight_ih_l0_reverse, lstm.weight_hh_l0_reverse, lstm.bias_ih_l0_reverse, lstm.bias_hh_l0_reverse, True)
    return torch.cat([yf, yr], dim=-1)


def text_encoder_plain(te, tokens, mask):
    # mirrors TextEncoder.forward without pack_padded_sequence
    x = te.embedding(tokens).transpose(1, 2)  # [B,emb,T]
    m = mask.unsqueeze(1)
    x = x.masked_fill(m, 0.0)
    for c in te.cnn:
        x = c(x)
        x = x.masked_fill(m, 0.0)
    x = x.transpose(1, 2)  # [B,T,C]
    x = fast_lstm(te.lstm, x)  # [B,T,2H]
    x = x.transpose(-1, -2)  # [B,C,T]
    x = x.masked_fill(m, 0.0)
    return x  # t_en [B,512,T]


def duration_encoder_plain(de, x, style, mask):
    # mirrors DurationEncoder.forward without pack_padded_sequence. x = d_en [B,C,T]
    masks = mask
    x = x.permute(2, 0, 1)  # [T,B,C]
    s = style.expand(x.shape[0], x.shape[1], -1)  # [T,B,sty]
    x = torch.cat([x, s], axis=-1)  # [T,B,C+sty]
    x = x.masked_fill(masks.unsqueeze(-1).transpose(0, 1), 0.0)
    x = x.transpose(0, 1).transpose(-1, -2)  # [B,C+sty,T]
    for block in de.lstms:
        if isinstance(block, AdaLayerNorm):
            x = block(x.transpose(-1, -2), style).transpose(-1, -2)
            x = torch.cat([x, s.permute(1, 2, 0)], axis=1)
            x = x.masked_fill(masks.unsqueeze(-1).transpose(-1, -2), 0.0)
        else:
            x = x.transpose(-1, -2)  # [B,T,C]
            x = fast_lstm(block, x)
            x = x.transpose(-1, -2)  # [B,C,T]
    return x.transpose(-1, -2)  # [B,T,C]


class EngineA(nn.Module):
    def __init__(self, model, num_steps):
        super().__init__()
        self.te = model.text_encoder
        self.bert = model.bert
        self.bert_encoder = model.bert_encoder
        self.predictor = model.predictor
        self.diff = DiffusionTRTExportModule(model.diffusion.diffusion, num_steps=num_steps)

    def _denoise32(self, x32, sigma32, embedding, features, scale):
        # heavy denoise transformer runs fp16; return fp32 for stable sampler arithmetic.
        # scale drives classifier-free guidance (traced with !=1 -> CFG branch baked in).
        out = self.diff.diffusion.denoise_fn(
            x32.to(embedding.dtype), sigmas=sigma32.expand(x32.shape[0]).to(embedding.dtype),
            embedding=embedding, features=features, embedding_scale=scale)
        return out.float()

    def _adpm2_step32(self, x, sigma, sigma_next, step_noise, embedding, features, scale):
        # ADPM2 step in fp32 (divisions by small sigma overflow in fp16)
        sigma_up = torch.sqrt(sigma_next ** 2 * (sigma ** 2 - sigma_next ** 2) / sigma ** 2)
        sigma_down = torch.sqrt(sigma_next ** 2 - sigma_up ** 2)
        sigma_mid = (sigma + sigma_down) / 2  # sampler_rho=1
        d = (x - self._denoise32(x, sigma, embedding, features, scale)) / sigma
        x_mid = x + d * (sigma_mid - sigma)
        d_mid = (x_mid - self._denoise32(x_mid, sigma_mid, embedding, features, scale)) / sigma_mid
        x = x + d_mid * (sigma_down - sigma)
        return x + step_noise * sigma_up

    def forward(self, tokens, mask, ref_s, noise, step_noise, alpha, beta, sigmas, scale):
        t_en = text_encoder_plain(self.te, tokens, mask)
        # bert stays fp32 (HF Albert LayerNorm is fp32-internal); cast to fp16 at boundary
        bert_dur = self.bert(tokens, attention_mask=(~mask).int()).to(t_en.dtype)
        d_en = self.bert_encoder(bert_dur).transpose(-1, -2)
        # inline the ADPM2 diffusion loop so each step output is a virtual split output,
        # breaking cross-step matmul fusion that crashes TRT enqueue with weight inputs.
        # `sigmas` is a runtime input: fewer diffusion steps = a schedule padded with
        # identity steps (sigma_next==sigma), so step count is dynamic up to compiled max.
        dsplits = []
        sigmas = sigmas.float()
        x = sigmas[0] * noise.float()
        for i in range(self.diff.num_steps - 1):
            x = self._adpm2_step32(x, sigmas[i], sigmas[i + 1], step_noise[:, i].float(), bert_dur, ref_s, scale)
            dsplits.append(x.to(t_en.dtype))
        s_pred = x.squeeze(1).to(t_en.dtype)
        s = s_pred[:, 128:]
        ref = s_pred[:, :128]
        ref = alpha * ref + (1 - alpha) * ref_s[:, :128]
        s = beta * s + (1 - beta) * ref_s[:, 128:]
        d = duration_encoder_plain(self.predictor.text_encoder, d_en, s, mask)
        x2 = fast_lstm(self.predictor.lstm, d)
        duration = self.predictor.duration_proj(x2)
        duration = torch.sigmoid(duration).sum(axis=-1)  # [B,T]
        # extra tensors double as virtual split outputs to break the weight-input
        # conv/matmul fusion that crashes TRT enqueue on large graphs.
        return (duration, d, t_en, s, ref, bert_dur, d_en, *dsplits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/workspace/tinfer/converted_models/agnieszka/model.pth")
    ap.add_argument("--voice", default="/workspace/tinfer/converted_models/agnieszka/voices/agnieszka-best.pth")
    ap.add_argument("--out", default="/workspace/tinfer/runtime/engines")
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--cfg", action="store_true", help="dynamic embedding_scale (classifier-free guidance, 2x diffusion cost)")
    ap.add_argument("--no-diff", action="store_true", help="diagnostic: skip diffusion, take s_pred as input (measures LSTM/bert cost)")
    ap.add_argument("--baked", action="store_true", help="diagnostic: bake weights (no weight-input) to test TRT LSTM speed")
    ap.add_argument("--fixed", action="store_true", help="static shapes (min=opt=max) -> CUDA-graph-capturable engine")
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)

    torch.manual_seed(1234)
    m = load_model(args.model)
    v = torch.load(args.voice, map_location="cpu", weights_only=True)
    net = m._model

    # capture reference front-end tensors from the real pipeline
    import tinfer.models.impl.styletts2.model.model as mm
    cap = {}
    orig = mm.StyleTTS2._run_model_forward
    def spy(self, tokens, ilen, mask, ref_s, prev, params, bs, il):
        cap.update(tokens=tokens.detach(), mask=mask.detach(), ref_s=ref_s.detach())
        return orig(self, tokens, ilen, mask, ref_s, prev, params, bs, il)
    mm.StyleTTS2._run_model_forward = spy
    m.load_voice_from_vector("v", v)
    m.generate_batch([TEXT_150], [{"voice_id": "v", "base_voice": v.numpy()}],
                     [{"use_diffusion": True, "diffusion_steps": args.steps}], [{"alignment_type": "none"}])
    mm.StyleTTS2._run_model_forward = orig
    tokens, mask, ref_s = cap["tokens"], cap["mask"], cap["ref_s"]
    B, T = tokens.shape
    print(f"captured tokens {tuple(tokens.shape)} mask {tuple(mask.shape)} ref_s {tuple(ref_s.shape)}")

    net = m._model
    a = EngineA(net, args.steps).eval()
    strip_weight_norm(a)
    a = a.half()
    a.bert = a.bert.float()  # keep PLBERT fp32 to avoid mixed-type LayerNorm at parse
    ref_s = ref_s.half()
    noise = torch.randn(B, 1, 256, device="cuda", dtype=torch.float16)
    step_noise = torch.randn(B, args.steps - 1, 1, 256, device="cuda", dtype=torch.float16)
    alpha = torch.full((B, 1), 0.3, device="cuda", dtype=torch.float16)
    beta = torch.full((B, 1), 0.7, device="cuda", dtype=torch.float16)
    sigmas = a.diff._sigmas(torch.device("cuda"), torch.float32)  # [steps+1]
    # scale=1.5 tensor bakes the CFG branch (dynamic scale input); float 1.0 = single-run fast path
    scale = torch.tensor([1.5], device="cuda", dtype=torch.float32) if args.cfg else 1.0
    with torch.no_grad():
        outs = a(tokens, mask, ref_s, noise, step_noise, alpha, beta, sigmas, scale)
    dur, d, t_en, s, ref = outs[:5]
    print(f"outputs: dur {tuple(dur.shape)} d {tuple(d.shape)} t_en {tuple(t_en.shape)} s {tuple(s.shape)} ref {tuple(ref.shape)}")
    if args.validate_only:
        return

    dstep_names = [f"dstep{i}" for i in range(args.steps - 1)]
    onames = ["dur", "d", "t_en", "s", "ref", "bert_dur", "d_en"] + dstep_names
    inames = ["tokens", "mask", "ref_s", "noise", "step_noise", "alpha", "beta", "sigmas"]
    if args.cfg:
        inames.append("scale")
    Tmax = 512
    dax = {"tokens": {0: "B", 1: "T"}, "mask": {0: "B", 1: "T"}, "ref_s": {0: "B"},
           "noise": {0: "B"}, "step_noise": {0: "B"}, "alpha": {0: "B"}, "beta": {0: "B"},
           "dur": {0: "B", 1: "T"}, "d": {0: "B", 1: "T"}, "t_en": {0: "B", 2: "T"},
           "s": {0: "B"}, "ref": {0: "B"}, "bert_dur": {0: "B", 1: "T"}, "d_en": {0: "B", 2: "T"}}
    for nm in dstep_names:
        dax[nm] = {0: "B"}
    K = args.steps
    prof = {
        "tokens": ((1, 8), (1, T), (16, Tmax)),
        "mask": ((1, 8), (1, T), (16, Tmax)),
        "ref_s": ((1, 256), (1, 256), (16, 256)),
        "noise": ((1, 1, 256), (1, 1, 256), (16, 1, 256)),
        "step_noise": ((1, K - 1, 1, 256), (1, K - 1, 1, 256), (16, K - 1, 1, 256)),
        "alpha": ((1, 1), (1, 1), (16, 1)),
        "beta": ((1, 1), (1, 1), (16, 1)),
        "sigmas": ((K + 1,), (K + 1,), (K + 1,)),
    }
    if args.cfg:
        prof["scale"] = ((1,), (1,), (1,))
    if os.environ.get("TINFER_FIX_TOKENS"):
        # benchmark mode: batch dynamic 1..16 but token count fixed at the captured T
        # (150 chars). Shrinks the profile hugely -> fast build, valid across batches.
        # opt batch must match the other inputs (all opt batch = 1) or TRT rejects the
        # profile ("dims named B must be equal"). Only the token dim is pinned to T.
        prof["tokens"] = ((1, T), (1, T), (16, T))
        prof["mask"] = ((1, T), (1, T), (16, T))
    if args.fixed:
        # static shapes (min==opt==max) so TRT unrolls the LSTMs into a CUDA-graph-
        # capturable static graph instead of a non-capturable dynamic Myelin loop.
        prof = {k: (v[1], v[1], v[1]) for k, v in prof.items()}
    example = (tokens, mask, ref_s, noise, step_noise, alpha, beta, sigmas, scale)
    engine_path, weights, wnames = export_promote_build(
        a, example, inames, onames, dax, prof, out, "A", strongly_typed=True, promote=not args.baked)
    print(f"built {engine_path}  ({len(wnames)} weight inputs)")
    r = WeightInputRunner(engine_path)
    wdump = {n: torch.as_tensor(weights[n]).to(r.dtype(n)) for n in wnames}
    dump_tensors(out / "A.weights", wdump)
    refd = {"tokens": tokens.long(), "mask": mask, "ref_s": ref_s,
            "noise": noise.half(), "step_noise": step_noise.half(),
            "alpha": alpha.half(), "beta": beta.half(), "sigmas": sigmas.float(),
            "dur": dur.half(), "t_en": t_en.half()}
    if args.cfg:
        refd["scale"] = scale.float()
    dump_tensors(out / "A.ref", refd)
    print(f"dumped A.weights and A.ref")


if __name__ == "__main__":
    main()
