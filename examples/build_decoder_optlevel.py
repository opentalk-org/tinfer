import sys, time, statistics
sys.path.insert(0, "/tmp/ort")
import numpy as np
import onnx
from onnx import numpy_helper, TensorProto
import tensorrt as trt

DIR = "/workspace/converted_models/libri/tensorrt"
SRC = f"{DIR}/decoder_dynamic_winput.onnx"     # ALL float params already promoted to inputs
logger = trt.Logger(trt.Logger.ERROR)
FLOATS = {TensorProto.FLOAT, TensorProto.FLOAT16}

# weight values from the original baked onnx
orig = onnx.load(f"{DIR}/decoder_dynamic.onnx")
weights_np = {i.name: numpy_helper.to_array(i) for i in orig.graph.initializer if i.data_type in FLOATS}
m = onnx.load(SRC)
wshapes = {vi.name: [d.dim_value for d in vi.type.tensor_type.shape.dim] for vi in m.graph.input}
ACT = {"asr","f0","noise","style","har"}
print(f"{len(weights_np)} weight tensors as inputs")

def act_shape(n,F):
    return {"asr":(1,512,F),"f0":(1,2*F),"noise":(1,2*F),"style":(1,128),"har":(1,1,600*F)}[n]

def try_build(optlevel):
    builder = trt.Builder(logger); network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    with open(SRC,"rb") as f:
        if not parser.parse(f.read()): return None, 0
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 6<<30)
    try: config.builder_optimization_level = optlevel
    except Exception as e: print("  optlevel set fail", e)
    prof = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        n = network.get_input(i).name
        if n in ACT: prof.set_shape(n, act_shape(n,128), act_shape(n,256), act_shape(n,512))
        else:
            s=tuple(wshapes[n]); prof.set_shape(n,s,s,s)
    config.add_optimization_profile(prof)
    t0=time.time(); plan=builder.build_serialized_network(network,config); return plan, time.time()-t0

plan=None; used=None
for lvl in [3,2,1,0]:
    print(f"trying builder_optimization_level={lvl} ...", flush=True)
    plan,bt = try_build(lvl)
    if plan is not None:
        used=lvl; print(f"  BUILD OK at level {lvl} in {bt:.1f}s, engine {plan.nbytes/1e6:.1f} MB"); break
    else:
        print(f"  failed at level {lvl}")
if plan is None:
    print("ALL OPT LEVELS FAILED"); sys.exit(1)

import torch
dev=torch.device("cuda")
eng=trt.Runtime(logger).deserialize_cuda_engine(bytes(plan)); ctx=eng.create_execution_context()
def tdt(n): return {trt.DataType.FLOAT:torch.float32,trt.DataType.HALF:torch.float16}[eng.get_tensor_dtype(n)]
wbuf={n:torch.tensor(v).to(dev).to(tdt(n)) for n,v in weights_np.items()}
onames=[eng.get_tensor_name(i) for i in range(eng.num_io_tensors) if eng.get_tensor_mode(eng.get_tensor_name(i))==trt.TensorIOMode.OUTPUT]
def run(F):
    acts={"asr":torch.randn(1,512,F,device=dev),"f0":torch.rand(1,2*F,device=dev)*120+80,
          "noise":torch.randn(1,2*F,device=dev),"style":torch.randn(1,128,device=dev),
          "har":torch.randn(1,1,600*F,device=dev)}
    for i in range(eng.num_io_tensors):
        n=eng.get_tensor_name(i)
        if eng.get_tensor_mode(n)==trt.TensorIOMode.INPUT:
            t=(wbuf[n] if n in wbuf else acts[n].to(tdt(n))).contiguous()
            ctx.set_input_shape(n,tuple(t.shape)); ctx.set_tensor_address(n,t.data_ptr())
    out=torch.empty((1,1,F*300),device=dev,dtype=tdt(onames[0])); ctx.set_tensor_address(onames[0],out.data_ptr())
    ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream); return out
def timeit(F,iters=50,warm=10):
    for _ in range(warm): run(F)
    torch.cuda.synchronize(); ts=[]
    for _ in range(iters):
        torch.cuda.synchronize(); e0=torch.cuda.Event(enable_timing=True); e1=torch.cuda.Event(enable_timing=True)
        e0.record(); run(F); e1.record(); torch.cuda.synchronize(); ts.append(e0.elapsed_time(e1))
    ts.sort(); return statistics.mean(ts)
ref={128:6.45,256:10.26,449:19.09}
print(f"\n(opt level {used})  {'frames':>6} | {'conv-w-input':>12} | baked TRT")
for F in [128,256,449]:
    ms=timeit(F); print(f"        {F:>6} | {ms:10.3f}ms | ~{ref[F]}ms -> {ms/ref[F]:.2f}x")
