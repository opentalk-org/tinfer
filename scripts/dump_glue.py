"""Dump a reference for validating the glue2 CUDA kernel (source module + STFT):
F0, l_linear weight/bias, and har = generator._preprocess_f0(F0) with randomness
disabled (so it matches the kernel run with randScale=0)."""
from __future__ import annotations
from pathlib import Path
import torch
import torch.nn as nn

from build_common import load_model, strip_weight_norm, TEXT_150
import tinfer.models.impl.styletts2.model.modules.istftnet as istftnet
from tinfer.models.impl.styletts2.model.trt3.core import dump_tensors

OUT = Path("/workspace/tinfer/runtime/engines")
torch.manual_seed(1234)
m = load_model("/workspace/tinfer/converted_models/agnieszka/model.pth")
v = torch.load("/workspace/tinfer/converted_models/agnieszka/voices/agnieszka-best.pth", map_location="cpu", weights_only=True)
dec = m._model.decoder
strip_weight_norm(dec)

cap = {}
orig = istftnet.Decoder.forward
def spy(self, asr, F0, N, s):
    cap["F0"] = F0.detach()
    return orig(self, asr, F0, N, s)
istftnet.Decoder.forward = spy
m.load_voice_from_vector("v", v)
m.generate_batch([TEXT_150], [{"voice_id": "v", "base_voice": v.numpy()}],
                 [{"use_diffusion": True, "diffusion_steps": 5}], [{"alignment_type": "none"}])
istftnet.Decoder.forward = orig

F0 = cap["F0"]  # [B, 2F]
gen = dec.generator.eval().float().cuda()
lin = gen.m_source.l_linear
linW = lin.weight.detach().reshape(-1)  # [9]
linB = lin.bias.detach().reshape(-1)    # [1]

# disable randomness to match kernel randScale=0
_rand, _randn_like = torch.rand, torch.randn_like
torch.rand = lambda *a, **k: torch.zeros(*a, **{kk: vv for kk, vv in k.items() if kk != "device"},
                                          device=k.get("device", "cuda"))
torch.randn_like = lambda x, *a, **k: torch.zeros_like(x)
try:
    with torch.no_grad():
        har = gen._preprocess_f0(F0.float().cuda()).detach()
finally:
    torch.rand, torch.randn_like = _rand, _randn_like

print("F0", tuple(F0.shape), "har", tuple(har.shape), "linW", tuple(linW.shape))
dump_tensors(OUT / "glue.ref", {"f0": F0.half(), "linW": linW.half(), "linB": linB.half(),
                                "har": har.half()})
print(f"dumped {OUT}/glue.ref")
