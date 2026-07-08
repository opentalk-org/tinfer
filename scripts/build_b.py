"""Build engine B = predictor.F0Ntrain(en, s) -> F0, N. Weights as inputs.
en is the frame-resolution prosody feature (d @ alignment) from glue1; s is the
128-d style. Dumps B.engine, B.weights, B.ref for the C++ runtime."""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from build_common import load_model, strip_weight_norm, export_promote_build, TEXT_150
import tinfer.models.impl.styletts2.model.modules.blocks as blocks
from tinfer.models.impl.styletts2.model.trt3.core import WeightInputRunner, dump_tensors


class EngineB(nn.Module):
    def __init__(self, predictor, nsplit):
        super().__init__()
        self.predictor = predictor
        self.nsplit = nsplit

    def forward(self, en, s):
        p = self.predictor
        x, _ = p.shared(en.transpose(-1, -2))
        splits = []
        F0 = x.transpose(-1, -2)
        for blk in p.F0:
            F0 = blk(F0, s)
            splits.append(F0)
        F0 = p.F0_proj(F0)
        N = x.transpose(-1, -2)
        for blk in p.N:
            N = blk(N, s)
            splits.append(N)
        N = p.N_proj(N)
        return (F0.squeeze(1), N.squeeze(1), *splits[:self.nsplit])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/workspace/tinfer/converted_models/agnieszka/model.pth")
    ap.add_argument("--voice", default="/workspace/tinfer/converted_models/agnieszka/voices/agnieszka-best.pth")
    ap.add_argument("--out", default="/workspace/tinfer/runtime/engines")
    ap.add_argument("--max-batch", type=int, default=16)
    ap.add_argument("--max-frames", type=int, default=1200)
    ap.add_argument("--nsplit", type=int, default=6)
    args = ap.parse_args()
    out = Path(args.out)

    torch.manual_seed(1234)
    m = load_model(args.model)
    v = torch.load(args.voice, map_location="cpu", weights_only=True)
    m.load_voice_from_vector("v", v)
    predictor = m._model.predictor
    strip_weight_norm(predictor)

    cap = {}
    orig = blocks.ProsodyPredictor.F0Ntrain
    def spy(self, x, s):
        out = orig(self, x, s)
        cap.update(en=x.detach(), s=s.detach(), F0=out[0].detach(), N=out[1].detach())
        return out
    blocks.ProsodyPredictor.F0Ntrain = spy
    m.generate_batch([TEXT_150], [{"voice_id": "v", "base_voice": v.numpy()}],
                     [{"use_diffusion": True, "diffusion_steps": 5}], [{"alignment_type": "none"}])
    blocks.ProsodyPredictor.F0Ntrain = orig

    en, s, F0ref, Nref = cap["en"], cap["s"], cap["F0"], cap["N"]
    C = en.shape[1]
    predictor = predictor.eval().half().cuda()
    en, s = en.half(), s.half()
    B, L = en.shape[0], en.shape[2]
    print(f"captured en {tuple(en.shape)} s {tuple(s.shape)} F0 {tuple(F0ref.shape)}  (C={C})")

    def sh(b, l):
        return {"en": (b, C, l), "s": (b, 128)}
    prof = {n: (sh(1, 32)[n], sh(min(8, B), L)[n], sh(args.max_batch, args.max_frames)[n])
            for n in ["en", "s"]}
    split_names = [f"split{i}" for i in range(args.nsplit)]
    onames = ["f0", "noise"] + split_names
    dax = {"en": {0: "B", 2: "L"}, "s": {0: "B"}, "f0": {0: "B", 1: "T"}, "noise": {0: "B", 1: "T"}}
    for nm in split_names:
        dax[nm] = {0: "B", 2: "S"}

    engine_path, weights, wnames = export_promote_build(
        EngineB(predictor, args.nsplit).eval(), (en, s),
        ["en", "s"], onames, dax, prof, out, "B")
    print(f"built {engine_path}  ({len(wnames)} weight inputs)")

    r = WeightInputRunner(engine_path)
    wdump = {n: torch.as_tensor(weights[n]).to(r.dtype(n)) for n in wnames}
    dump_tensors(out / "B.weights", wdump)
    dump_tensors(out / "B.ref", {"en": en, "s": s, "f0": F0ref.half(), "noise": Nref.half()})
    print(f"dumped {out}/B.weights and {out}/B.ref")


if __name__ == "__main__":
    main()
