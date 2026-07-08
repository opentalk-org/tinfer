import sys, statistics
from pathlib import Path
import torch
sys.path.insert(0, "/workspace/tinfer/tinfer")
from tinfer.models.impl.styletts2.model.modules import tensorrt_runtime as rt

ENGINE_DIR = Path("/workspace/converted_models/libri/tensorrt")
dev = torch.device("cuda")
dec = rt.get_tensorrt_decoder_runner(ENGINE_DIR)

def mk(F):
    return {
        "asr": torch.randn(1,512,F,device=dev,dtype=torch.float16),
        "f0": (torch.rand(1,2*F,device=dev,dtype=torch.float16)*120+80),
        "noise": torch.randn(1,2*F,device=dev,dtype=torch.float16),
        "style": torch.randn(1,128,device=dev,dtype=torch.float16),
        "har": torch.randn(1,1,600*F,device=dev,dtype=torch.float16),
    }

def timeit(F, iters=50, warm=10):
    o = dec.run(mk(F)); sh = tuple(o["audio"].shape)
    for _ in range(warm): dec.run(mk(F))
    torch.cuda.synchronize()
    ts=[]
    for _ in range(iters):
        inp=mk(F); torch.cuda.synchronize()
        e0=torch.cuda.Event(enable_timing=True); e1=torch.cuda.Event(enable_timing=True)
        e0.record(); dec.run(inp); e1.record(); torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1))
    ts.sort(); return statistics.mean(ts), ts[len(ts)//2], min(ts), max(ts), sh

print("=== DECODER engine (FP16) via shared runner (dedicated stream) ===")
for F in [128,256,449,512]:
    try:
        mean,p50,mn,mx,sh = timeit(F)
        print(f" F={F:4d} out{sh} ({F*300/24000:5.2f}s): mean {mean:6.3f} ms p50 {p50:6.3f} min {mn:6.3f} max {mx:6.3f}")
    except Exception as e:
        print(f" F={F:4d}: ERROR {type(e).__name__}: {e}")
