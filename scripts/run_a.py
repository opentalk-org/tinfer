"""Standalone engine-A runner: binds weights + activations, runs with overridable
alpha / beta / sigmas (diffusion schedule) / scale (CFG), returns outputs. Used to
verify those inputs are runtime-dynamic (change output without an engine rebuild)."""
import struct, sys
import numpy as np
import torch
import tensorrt as trt

import os
ENG = sys.argv[1] if len(sys.argv) > 1 else "/workspace/tinfer/runtime/engines/A.engine"
DIR = os.path.dirname(ENG)
CODE2NP = {0: np.float16, 1: np.float32, 2: np.int32, 3: np.int64, 4: np.bool_}
TRT2T = {trt.DataType.HALF: torch.float16, trt.DataType.FLOAT: torch.float32,
         trt.DataType.INT32: torch.int32, trt.DataType.INT64: torch.int64, trt.DataType.BOOL: torch.bool}


def load_tinf(path):
    f = open(path, "rb"); assert f.read(4) == b"TINF"
    n = struct.unpack("<i", f.read(4))[0]; out = {}
    for _ in range(n):
        nl = struct.unpack("<i", f.read(4))[0]; name = f.read(nl).decode()
        code = struct.unpack("<i", f.read(4))[0]; nd = struct.unpack("<i", f.read(4))[0]
        dims = [struct.unpack("<q", f.read(8))[0] for _ in range(nd)]
        cnt = int(np.prod(dims)) if dims else 1
        arr = np.frombuffer(f.read(cnt * np.dtype(CODE2NP[code]).itemsize), dtype=CODE2NP[code]).reshape(dims)
        out[name] = torch.from_numpy(arr.copy())
    return out


logger = trt.Logger(trt.Logger.ERROR)
rt = trt.Runtime(logger)
eng = rt.deserialize_cuda_engine(open(ENG, "rb").read())
ctx = eng.create_execution_context()
W = load_tinf(f"{DIR}/A.weights"); R = load_tinf(f"{DIR}/A.ref")
names = [eng.get_tensor_name(i) for i in range(eng.num_io_tensors)]
INS = [n for n in names if eng.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
OUTS = [n for n in names if eng.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
HAS_SCALE = "scale" in INS


def run(alpha=None, beta=None, sigmas=None, scale=None):
    keep = []
    for n in INS:
        if n in W:
            t = W[n]
        elif n == "alpha" and alpha is not None:
            t = torch.full((1, 1), float(alpha), dtype=torch.float16)
        elif n == "beta" and beta is not None:
            t = torch.full((1, 1), float(beta), dtype=torch.float16)
        elif n == "sigmas" and sigmas is not None:
            t = torch.tensor(sigmas, dtype=torch.float32)
        elif n == "scale" and scale is not None:
            t = torch.tensor([float(scale)], dtype=torch.float32)
        else:
            t = R[n]
        t = t.cuda().contiguous(); keep.append(t)
        ctx.set_input_shape(n, tuple(t.shape))
        ctx.set_tensor_address(n, t.data_ptr())
    outs = {}
    for n in OUTS:
        shp = tuple(ctx.get_tensor_shape(n))
        b = torch.empty(shp, dtype=TRT2T[eng.get_tensor_dtype(n)], device="cuda")
        outs[n] = b; ctx.set_tensor_address(n, b.data_ptr())
    ctx.execute_async_v3(0); torch.cuda.synchronize()
    return outs


def summ(o):
    return dict(dur_sum=float(o["dur"].float().sum()),
                s_l2=float(o["s"].float().norm()), ref_l2=float(o["ref"].float().norm()))


if __name__ == "__main__":
    sig5 = R["sigmas"].tolist()
    print("engine inputs:", [n for n in INS if n in
          ("tokens", "mask", "ref_s", "noise", "step_noise", "alpha", "beta", "sigmas", "scale")])
    print("HAS_SCALE(cfg):", HAS_SCALE)
    print("baseline sigmas(5-step):", [round(x, 3) for x in sig5])
    base = summ(run())
    print("baseline            ", base)
    print("alpha=0.0           ", summ(run(alpha=0.0)))
    print("alpha=1.0           ", summ(run(alpha=1.0)))
    print("beta=0.0            ", summ(run(beta=0.0)))
    print("beta=1.0            ", summ(run(beta=1.0)))
    # fewer effective steps: keep schedule length, make trailing transitions identity
    # (sigma_next==sigma -> no-op ADPM2 step). Same length as baseline (dynamic values).
    sig2 = [sig5[0], sig5[1], sig5[1], sig5[1], sig5[1], sig5[1]]  # ~1 real step
    sig3 = [sig5[0], sig5[1], sig5[2], sig5[2], sig5[2], sig5[2]]  # ~2 real steps
    print("sigmas ~1-step (ident-pad)", summ(run(sigmas=sig2)))
    print("sigmas ~2-step (ident-pad)", summ(run(sigmas=sig3)))
    if HAS_SCALE:
        print("scale=1.0           ", summ(run(scale=1.0)))
        print("scale=3.0           ", summ(run(scale=3.0)))
