"""Build engine C = decoder backbone + generator (forward_with_har), weights as
inputs, with a virtual output at the backbone/generator junction to avoid the TRT
conv-fusion enqueue crash. Dumps C.engine, C.weights, C.ref for the C++ runtime."""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from build_common import load_model, strip_weight_norm, export_promote_build, TEXT_150
import tinfer.models.impl.styletts2.model.modules.istftnet as istftnet
from tinfer.models.impl.styletts2.model.modules.decoder_blocks import DecoderBackbone
from tinfer.models.impl.styletts2.model.trt3.core import WeightInputRunner, dump_tensors


class EngineC(nn.Module):
    """decoder backbone + generator (forward_with_har) with intermediate tensors
    exposed as virtual outputs. These break the aggressive TensorRT conv fusion
    that segfaults enqueueV3 when all weights are runtime inputs; more virtual
    outputs are needed at higher builder optimization levels."""
    def __init__(self, dec, nsplit: int):
        super().__init__()
        self.dec = dec
        self.nsplit = nsplit

    def forward(self, asr, f0, noise, style, har):
        import torch.nn.functional as F
        import tinfer.models.impl.styletts2.model.modules.istftnet as istft
        gen = self.dec.generator
        x, _ = DecoderBackbone.forward(self.dec, asr, f0, noise, style)
        splits = [x]
        for i in range(gen.num_upsamples):
            x = F.leaky_relu(x, istft.LRELU_SLOPE)
            x_source = gen.noise_res[i](gen.noise_convs[i](har), style)
            splits.append(x_source)
            x = gen.ups[i](x)
            if i == gen.num_upsamples - 1:
                x = gen.reflection_pad(x)
            splits.append(x)
            x = x + x_source
            x = gen._process_resblocks(x, style, i)
            splits.append(x)
        audio = gen._generate_output(x)
        return (audio, *splits[:self.nsplit])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/workspace/tinfer/converted_models/agnieszka/model.pth")
    ap.add_argument("--voice", default="/workspace/tinfer/converted_models/agnieszka/voices/agnieszka-best.pth")
    ap.add_argument("--out", default="/workspace/tinfer/runtime/engines")
    ap.add_argument("--max-batch", type=int, default=16)
    ap.add_argument("--max-frames", type=int, default=1200)
    ap.add_argument("--nsplit", type=int, default=7)
    ap.add_argument("--baked", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)

    torch.manual_seed(1234)  # deterministic diffusion -> deterministic captured frame count
    m = load_model(args.model)
    v = torch.load(args.voice, map_location="cpu", weights_only=True)
    m.load_voice_from_vector("v", v)
    dec = m._model.decoder
    strip_weight_norm(dec)

    cap = {}
    orig = istftnet.Decoder.forward
    def spy(self, asr, F0, N, s):
        out = orig(self, asr, F0, N, s)
        cap.update(asr=asr.detach(), F0=F0.detach(), N=N.detach(), s=s.detach(), out=out.detach())
        return out
    istftnet.Decoder.forward = spy
    m.generate_batch([TEXT_150], [{"voice_id": "v", "base_voice": v.numpy()}],
                     [{"use_diffusion": True, "diffusion_steps": 5}], [{"alignment_type": "none"}])
    istftnet.Decoder.forward = orig

    asr, F0, N, s, ref = cap["asr"], cap["F0"], cap["N"], cap["s"], cap["out"]
    har = dec.generator._preprocess_f0(F0).detach()
    dec = dec.eval().half().cuda()
    asr, F0, N, s, har = asr.half(), F0.half(), N.half(), s.half(), har.half()
    B, L = asr.shape[0], asr.shape[2]
    print(f"captured B={B} L={L}  audio {tuple(ref.shape)}")

    def sh(b, l):
        return {"asr": (b, 512, l), "f0": (b, l * 2), "noise": (b, l * 2), "style": (b, 128),
                "har": (b, 22, l * 120 + 1)}
    mb, mf = args.max_batch, args.max_frames
    prof = {n: (sh(1, 128)[n], sh(min(8, B), L)[n], sh(mb, mf)[n])
            for n in ["asr", "f0", "noise", "style", "har"]}
    split_names = [f"split{i}" for i in range(args.nsplit)]
    onames = ["audio"] + split_names
    dax = {"asr": {0: "B", 2: "L"}, "f0": {0: "B", 1: "F"}, "noise": {0: "B", 1: "F"},
           "style": {0: "B"}, "har": {0: "B", 2: "H"}, "audio": {0: "B", 2: "T"}}
    for nm in split_names:
        dax[nm] = {0: "B", 2: "S"}

    engine_path, weights, wnames = export_promote_build(
        EngineC(dec, args.nsplit).eval(), (asr, F0, N, s, har),
        ["asr", "f0", "noise", "style", "har"], onames, dax, prof, out, "C",
        promote=not args.baked)
    print(f"built {engine_path}  ({len(wnames)} weight inputs)")

    # Dump weights + reference FIRST (before any enqueue), so the C++ runtime always
    # gets fresh artifacts even if the python-side validation enqueue trips a TRT bug.
    r = WeightInputRunner(engine_path)
    wdump = {n: torch.as_tensor(weights[n]).to(r.dtype(n)) for n in wnames}
    non_f16 = [n for n in wnames if r.dtype(n) != torch.float16]
    print(f"weight inputs not fp16: {len(non_f16)} {non_f16[:6]}")
    dump_tensors(out / "C.weights", wdump)
    dump_tensors(out / "C.ref", {"asr": asr, "f0": F0, "noise": N, "style": s, "har": har, "audio": ref.half()})
    print(f"dumped {out}/C.weights and {out}/C.ref  (validate in C++)")


if __name__ == "__main__":
    main()
