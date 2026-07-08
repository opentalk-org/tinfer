import sys, statistics
sys.path.insert(0, "/tmp/ort")
import numpy as np, onnx
from onnx import numpy_helper, TensorProto
import tensorrt as trt, torch

DIR = "/workspace/converted_models/libri/tensorrt"
logger = trt.Logger(trt.Logger.ERROR)
dev = torch.device("cuda")
FLOATS = {TensorProto.FLOAT, TensorProto.FLOAT16}
orig = onnx.load(f"{DIR}/decoder_dynamic.onnx")
weights_np = {i.name: numpy_helper.to_array(i) for i in orig.graph.initializer if i.data_type in FLOATS}

with open("/tmp/dec_barrier.plan","rb") as f: plan = f.read()
eng = trt.Runtime(logger).deserialize_cuda_engine(plan)
ctx = eng.create_execution_context()
def tdt(n): return {trt.DataType.FLOAT:torch.float32, trt.DataType.HALF:torch.float16}[eng.get_tensor_dtype(n)]
wbuf = {n: torch.tensor(v).to(dev).to(tdt(n)) for n,v in weights_np.items()}
innames  = [eng.get_tensor_name(i) for i in range(eng.num_io_tensors) if eng.get_tensor_mode(eng.get_tensor_name(i))==trt.TensorIOMode.INPUT]
outnames = [eng.get_tensor_name(i) for i in range(eng.num_io_tensors) if eng.get_tensor_mode(eng.get_tensor_name(i))==trt.TensorIOMode.OUTPUT]
ACT = {"asr","f0","noise","style","har"}
print("outputs:", outnames)

def one(F, diag=False):
    acts = {"asr":torch.randn(1,512,F,device=dev,dtype=torch.float16),
            "f0":(torch.rand(1,2*F,device=dev,dtype=torch.float16)*120+80),
            "noise":torch.randn(1,2*F,device=dev,dtype=torch.float16),
            "style":torch.randn(1,128,device=dev,dtype=torch.float16),
            "har":torch.randn(1,1,600*F,device=dev,dtype=torch.float16)}
    for n in innames:
        t = (wbuf[n] if n in wbuf else acts[n]).contiguous()
        ctx.set_input_shape(n, tuple(t.shape)); ctx.set_tensor_address(n, t.data_ptr())
    outs = []
    for n in outnames:
        sh = tuple(ctx.get_tensor_shape(n))
        if diag: print("  out", n, sh)
        assert all(d >= 0 for d in sh), (n, sh)
        o = torch.empty(sh, device=dev, dtype=tdt(n)); outs.append(o)
        ctx.set_tensor_address(n, o.data_ptr())
    e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(); e0.record()
    ok = ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    e1.record(); torch.cuda.synchronize()   # sync BEFORE acts/outs freed
    if not ok: raise RuntimeError("exec failed")
    return e0.elapsed_time(e1), [o for o in outs if o is outs[0]]  # keep audio ref

print("warmup + shape check:")
one(128, diag=True)
ref = {128:6.45, 256:10.26, 449:19.09}
print(f"\n{'frames':>6} | {'conv-w-input':>12} | baked | ratio")
for F in [128,256,449]:
    for _ in range(10): one(F)
    ts = sorted(one(F)[0] for _ in range(50))
    ms = statistics.mean(ts)
    print(f"{F:>6} | {ms:10.3f}ms | ~{ref[F]:5.2f} | {ms/ref[F]:.2f}x")
