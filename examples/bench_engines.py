import sys, statistics
from pathlib import Path
import torch

sys.path.insert(0, "/workspace/tinfer/tinfer")
from tinfer.models.impl.styletts2.model.modules import tensorrt_runtime as rt

ENGINE_DIR = Path("/workspace/converted_models/libri/tensorrt")
dev = torch.device("cuda")

def time_engine(runner, make_inputs, iters=50, warm=10):
    for _ in range(warm):
        runner.run(make_inputs())
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        inp = make_inputs()
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        runner.run(inp)
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1))
    ts.sort()
    return statistics.mean(ts), ts[len(ts)//2], min(ts), max(ts)

print("Loading engines...")
dec = rt.get_tensorrt_decoder_runner(ENGINE_DIR)
dif = rt.get_tensorrt_diffusion_runner(ENGINE_DIR, num_steps=5)
print("decoder inputs:", dec.input_names)
print("diffusion inputs:", dif.input_names)

def run_decoder_manual(runner, inputs, out_samples):
    ctx = runner._context
    for name in runner._input_names:
        t = inputs[name].contiguous()
        ctx.set_input_shape(name, tuple(t.shape))
        ctx.set_tensor_address(name, t.data_ptr())
    oname = runner._output_names[0]
    out = torch.empty((1,1,out_samples), device=dev, dtype=torch.float16)
    ctx.set_tensor_address(oname, out.data_ptr())
    s = torch.cuda.current_stream()
    ok = ctx.execute_async_v3(s.cuda_stream)
    if not ok: raise RuntimeError("decoder exec failed")
    return out

def time_decoder(F, iters=50, warm=10):
    def mk():
        return {
            "asr": torch.randn(1,512,F,device=dev,dtype=torch.float16),
            "f0": (torch.rand(1,2*F,device=dev,dtype=torch.float16)*120+80),
            "noise": torch.randn(1,2*F,device=dev,dtype=torch.float16),
            "style": torch.randn(1,128,device=dev,dtype=torch.float16),
            "har": torch.randn(1,22,F*120+1,device=dev,dtype=torch.float16),
        }
    out_samples = F*300
    for _ in range(warm):
        run_decoder_manual(dec, mk(), out_samples)
    torch.cuda.synchronize()
    ts=[]
    for _ in range(iters):
        inp=mk(); torch.cuda.synchronize()
        e0=torch.cuda.Event(enable_timing=True); e1=torch.cuda.Event(enable_timing=True)
        e0.record(); run_decoder_manual(dec, inp, out_samples); e1.record()
        torch.cuda.synchronize(); ts.append(e0.elapsed_time(e1))
    ts.sort(); return statistics.mean(ts), ts[len(ts)//2], min(ts), max(ts)

print("\n" + "="*70)
print("DECODER engine (FP16)")
print("="*70)
for F in [128, 256, 449, 512]:
    try:
        mean,p50,mn,mx = time_decoder(F)
        print(f" asr_frames={F:4d} ({F*300/24000:5.2f}s audio) : mean {mean:6.3f} ms  p50 {p50:6.3f}  min {mn:6.3f}  max {mx:6.3f}")
    except Exception as e:
        print(f" asr_frames={F:4d}: ERROR {e}")

print("\n" + "="*70)
print("DIFFUSION engine (FP16, 5 steps)")
print("="*70)
for T in [64, 128, 171, 256]:
    def mk(T=T):
        return {
            "noise": torch.randn(1,1,256,device=dev,dtype=torch.float16),
            "step_noise": torch.randn(1,4,1,256,device=dev,dtype=torch.float16),
            "embedding": torch.randn(1,T,768,device=dev,dtype=torch.float16),
            "features": torch.randn(1,256,device=dev,dtype=torch.float16),
        }
    try:
        mean,p50,mn,mx = time_engine(dif, mk)
        print(f" tokens={T:4d} : mean {mean:6.3f} ms  p50 {p50:6.3f}  min {mn:6.3f}  max {mx:6.3f}")
    except Exception as e:
        print(f" tokens={T:4d}: ERROR {e}")
