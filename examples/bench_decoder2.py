import sys, statistics
from pathlib import Path
import torch, tensorrt as trt

sys.path.insert(0, "/workspace/tinfer/tinfer")
from tinfer.models.impl.styletts2.model.modules import tensorrt_runtime as rt

ENGINE_DIR = Path("/workspace/converted_models/libri/tensorrt")
dev = torch.device("cuda")
dec = rt.get_tensorrt_decoder_runner(ENGINE_DIR)
eng = dec._engine

# introspect profile shapes
prof = {}
for name in dec.input_names:
    mn, opt, mx = eng.get_tensor_profile_shape(name, 0)
    prof[name] = (tuple(mn), tuple(opt), tuple(mx))
    print(f"{name:8s} min{tuple(mn)} opt{tuple(opt)} max{tuple(mx)} dtype={eng.get_tensor_dtype(name)}")
outname = dec.output_names[0]
print("output:", outname, eng.get_tensor_dtype(outname))

# asr opt frames -> relationship. asr shape (b,512,Fopt). har opt frames = prof['har'][1][-1]
asr_opt_F = prof["asr"][1][-1]
har_opt = prof["har"][1][-1]
har_ch = prof["har"][1][1]
print(f"\nasr_opt_F={asr_opt_F} har_opt_frames={har_opt} har_channels={har_ch} ratio={har_opt/asr_opt_F:.2f}")

def dt(name):
    return rt._torch_dtype_from_trt(eng.get_tensor_dtype(name))

def run_manual(F):
    har_frames = round(har_opt * F / asr_opt_F)
    out_samples = F*300
    inputs = {
        "asr": torch.randn(1,512,F,device=dev,dtype=dt("asr")),
        "f0": (torch.rand(1,2*F,device=dev,dtype=dt("f0"))*120+80),
        "noise": torch.randn(1,2*F,device=dev,dtype=dt("noise")),
        "style": torch.randn(1,128,device=dev,dtype=dt("style")),
        "har": torch.randn(1,har_ch,har_frames,device=dev,dtype=dt("har")),
    }
    ctx = dec._context
    for n in dec.input_names:
        t = inputs[n].contiguous(); ctx.set_input_shape(n, tuple(t.shape)); ctx.set_tensor_address(n, t.data_ptr())
    oshape = tuple(ctx.get_tensor_shape(outname))
    if any(d<0 for d in oshape):
        ctx.infer_shapes(); oshape = tuple(ctx.get_tensor_shape(outname))
    out = torch.empty(oshape if all(d>0 for d in oshape) else (1,1,out_samples), device=dev, dtype=dt(outname))
    ctx.set_tensor_address(outname, out.data_ptr())
    ok = ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    if not ok: raise RuntimeError("exec failed")
    return out, tuple(out.shape)

def timeit(F, iters=50, warm=10):
    o,sh = run_manual(F)
    for _ in range(warm): run_manual(F)
    torch.cuda.synchronize()
    ts=[]
    for _ in range(iters):
        torch.cuda.synchronize()
        e0=torch.cuda.Event(enable_timing=True); e1=torch.cuda.Event(enable_timing=True)
        e0.record(); run_manual(F); e1.record(); torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1))
    ts.sort(); return statistics.mean(ts), ts[len(ts)//2], min(ts), max(ts), sh

print("\n=== DECODER engine (FP16) pure ===")
for F in [128,256,449,512]:
    try:
        mean,p50,mn,mx,sh = timeit(F)
        print(f" F={F:4d} out{sh} ({F*300/24000:5.2f}s): mean {mean:6.3f} ms p50 {p50:6.3f} min {mn:6.3f} max {mx:6.3f}")
    except Exception as e:
        print(f" F={F:4d}: ERROR {e}")
