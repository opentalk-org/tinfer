import sys, time, statistics
sys.path.insert(0, "/tmp/ort")
import numpy as np
import onnx
from onnx import numpy_helper, TensorProto
import tensorrt as trt

DIR = "/workspace/converted_models/libri/tensorrt"
SRC = f"{DIR}/diffusion_dynamic_s5_winput.onnx"   # produced by previous script
logger = trt.Logger(trt.Logger.WARNING)

# read weight input shapes from onnx
m = onnx.load(SRC)
init_names = {i.name for i in m.graph.initializer}
weight_shapes = {}
for vi in m.graph.input:
    name = vi.name
    dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
    weight_shapes[name] = dims
ACT = {"noise","step_noise","embedding","features"}
weights_np = {}
# rebuild weight values from the ORIGINAL onnx initializers
orig = onnx.load(f"{DIR}/diffusion_dynamic_s5.onnx")
for init in orig.graph.initializer:
    if init.data_type in (TensorProto.FLOAT, TensorProto.FLOAT16):
        weights_np[init.name] = numpy_helper.to_array(init)

print(f"building TRT engine from weights-as-input ONNX ({len(weights_np)} weight inputs)...")
builder = trt.Builder(logger)
network = builder.create_network(0)
parser = trt.OnnxParser(network, logger)
with open(SRC,"rb") as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors): print("  parse err:", parser.get_error(i))
        sys.exit(1)
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4<<30)
for fl in ("FP16","kFP16"):
    if hasattr(trt.BuilderFlag, fl):
        config.set_flag(getattr(trt.BuilderFlag, fl)); print("set flag", fl); break
else:
    print("no FP16 flag; network is natively fp16")
prof = builder.create_optimization_profile()
def act_shape(name, T):
    return {"noise":(1,1,256),"step_noise":(1,4,1,256),"embedding":(1,T,768),"features":(1,256)}[name]
for i in range(network.num_inputs):
    inp = network.get_input(i); name = inp.name
    if name in ACT:
        prof.set_shape(name, act_shape(name,8), act_shape(name,128), act_shape(name,512))
    else:
        s = tuple(weight_shapes[name]); prof.set_shape(name, s, s, s)  # weights fixed
config.add_optimization_profile(prof)
t0=time.time()
plan = builder.build_serialized_network(network, config)
build_s = time.time()-t0
if plan is None:
    print("BUILD FAILED (TRT cannot make this graph with dynamic weights)"); sys.exit(1)
print(f"BUILD OK in {build_s:.1f}s, engine {plan.nbytes/1e6:.1f} MB")

eng = trt.Runtime(logger).deserialize_cuda_engine(bytes(plan))
ctx = eng.create_execution_context()
import torch
dev = torch.device("cuda")
def tdt(name):
    d = eng.get_tensor_dtype(name)
    return {trt.DataType.FLOAT:torch.float32, trt.DataType.HALF:torch.float16}[d]

# upload weights to GPU once (this is the ONE-TIME swap cost)
wbuf = {n: torch.tensor(v).to(dev).to(tdt(n)) for n,v in weights_np.items()}
onames = [eng.get_tensor_name(i) for i in range(eng.num_io_tensors)
          if eng.get_tensor_mode(eng.get_tensor_name(i))==trt.TensorIOMode.OUTPUT]

def run(T):
    acts = {"noise":torch.randn(1,1,256,device=dev),
            "step_noise":torch.randn(1,4,1,256,device=dev),
            "embedding":torch.randn(1,T,768,device=dev),
            "features":torch.randn(1,256,device=dev)}
    for i in range(eng.num_io_tensors):
        n = eng.get_tensor_name(i)
        if eng.get_tensor_mode(n)==trt.TensorIOMode.INPUT:
            t = (wbuf[n] if n in wbuf else acts[n].to(tdt(n))).contiguous()
            ctx.set_input_shape(n, tuple(t.shape)); ctx.set_tensor_address(n, t.data_ptr())
    outs={}
    for n in onames:
        sh=tuple(ctx.get_tensor_shape(n)); o=torch.empty(sh,device=dev,dtype=tdt(n))
        outs[n]=o; ctx.set_tensor_address(n,o.data_ptr())
    ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    return outs

def timeit(T,iters=50,warm=10):
    for _ in range(warm): run(T)
    torch.cuda.synchronize(); ts=[]
    for _ in range(iters):
        torch.cuda.synchronize()
        e0=torch.cuda.Event(enable_timing=True); e1=torch.cuda.Event(enable_timing=True)
        e0.record(); run(T); e1.record(); torch.cuda.synchronize(); ts.append(e0.elapsed_time(e1))
    ts.sort(); return statistics.mean(ts), ts[len(ts)//2], min(ts)

print(f"\n{'tokens':>7} | {'TRT w-input':>11} | (baked TRT ref)")
ref={128:0.92,171:1.10,256:1.26}
for T in [128,171,256]:
    mean,p50,mn=timeit(T)
    print(f"{T:>7} | {mean:9.3f}ms | ~{ref[T]}ms  -> {mean/ref[T]:.2f}x baked")
