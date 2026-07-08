import sys, time, statistics
sys.path.insert(0, "/tmp/ort")
import numpy as np
import onnx
from onnx import numpy_helper, TensorProto
import tensorrt as trt

DIR = "/workspace/converted_models/libri/tensorrt"
SRC = f"{DIR}/decoder_dynamic_winput.onnx"
logger = trt.Logger(trt.Logger.WARNING)

m = onnx.load(SRC)
weight_shapes = {vi.name:[d.dim_value for d in vi.type.tensor_type.shape.dim] for vi in m.graph.input}
ACT = {"asr","f0","noise","style","har"}
weights_np = {}
orig = onnx.load(f"{DIR}/decoder_dynamic.onnx")
for init in orig.graph.initializer:
    if init.data_type in (TensorProto.FLOAT, TensorProto.FLOAT16):
        weights_np[init.name] = numpy_helper.to_array(init)

print(f"building TRT DECODER engine, weights-as-input ({len(weights_np)} weight inputs)...")
builder = trt.Builder(logger); network = builder.create_network(0)
parser = trt.OnnxParser(network, logger)
with open(SRC,"rb") as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors): print("  parse err:", parser.get_error(i))
        sys.exit(2)
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 6<<30)
def act_shape(name,F):
    return {"asr":(1,512,F),"f0":(1,2*F),"noise":(1,2*F),"style":(1,128),"har":(1,1,600*F)}[name]
prof = builder.create_optimization_profile()
for i in range(network.num_inputs):
    n = network.get_input(i).name
    if n in ACT:
        prof.set_shape(n, act_shape(n,128), act_shape(n,256), act_shape(n,512))
    else:
        s=tuple(weight_shapes[n]); prof.set_shape(n,s,s,s)
config.add_optimization_profile(prof)
t0=time.time(); plan=builder.build_serialized_network(network,config); bs=time.time()-t0
if plan is None:
    print(f"BUILD FAILED after {bs:.1f}s (TRT cannot make conv-with-dynamic-weights decoder)"); sys.exit(1)
print(f"BUILD OK in {bs:.1f}s, engine {plan.nbytes/1e6:.1f} MB (baked decoder engine = 124.3 MB)")

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
print(f"\n{'frames':>7} | {'TRT w-input':>11} | baked TRT")
for F in [128,256,449]:
    print(f"{F:>7} | {timeit(F):9.3f}ms | ~{ref[F]}ms -> {timeit(F)/ref[F]:.2f}x")
