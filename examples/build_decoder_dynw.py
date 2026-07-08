import sys, time, statistics
sys.path.insert(0, "/tmp/ort")
import numpy as np
import onnx
from onnx import numpy_helper, helper, TensorProto
import tensorrt as trt

DIR = "/workspace/converted_models/libri/tensorrt"
SRC = f"{DIR}/decoder_dynamic.onnx"
DST = f"{DIR}/decoder_dynamic_convw.onnx"
FLOATS = {TensorProto.FLOAT, TensorProto.FLOAT16}
logger = trt.Logger(trt.Logger.ERROR)

# ---- promote ONLY conv/deconv/matmul weight+bias initializers ----
m = onnx.load(SRC); g = m.graph
init_map = {i.name: i for i in g.initializer}
wnames = set()
for node in g.node:
    if node.op_type in ("Conv", "ConvTranspose"):
        for idx in (1, 2):
            if idx < len(node.input) and node.input[idx] in init_map:
                wnames.add(node.input[idx])
    elif node.op_type in ("Gemm", "MatMul"):
        for nm in node.input:
            if nm in init_map:
                wnames.add(nm)
promote = [n for n in wnames if init_map[n].data_type in FLOATS]
weights = {n: numpy_helper.to_array(init_map[n]) for n in promote}
kept = [i for i in g.initializer if i.name not in set(promote)]
n_kept_float = sum(1 for i in kept if i.data_type in FLOATS)
del g.initializer[:]; g.initializer.extend(kept)
for n in promote:
    init = init_map[n]
    g.input.append(helper.make_tensor_value_info(n, init.data_type, list(init.dims)))
onnx.save(m, DST)
wbytes = sum(w.nbytes for w in weights.values())
print(f"promoted {len(promote)} conv/matmul weight tensors ({wbytes/1e6:.1f} MB); "
      f"kept {n_kept_float} small float params (alphas/norm/bias-broadcast) BAKED")

# ---- build TRT engine ----
ACT = {"asr", "f0", "noise", "style", "har"}
builder = trt.Builder(logger); network = builder.create_network(0)
parser = trt.OnnxParser(network, logger)
with open(DST, "rb") as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors): print("  parse err:", parser.get_error(i))
        sys.exit(2)
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 6 << 30)
wshapes = {vi.name: [d.dim_value for d in vi.type.tensor_type.shape.dim] for vi in g.input}
def act_shape(n, F):
    return {"asr": (1,512,F), "f0": (1,2*F), "noise": (1,2*F), "style": (1,128), "har": (1,1,600*F)}[n]
prof = builder.create_optimization_profile()
for i in range(network.num_inputs):
    n = network.get_input(i).name
    if n in ACT:
        prof.set_shape(n, act_shape(n,128), act_shape(n,256), act_shape(n,512))
    else:
        s = tuple(wshapes[n]); prof.set_shape(n, s, s, s)
config.add_optimization_profile(prof)
t0 = time.time(); plan = builder.build_serialized_network(network, config); bt = time.time()-t0
if plan is None:
    print(f"BUILD FAILED after {bt:.1f}s"); sys.exit(1)
print(f"BUILD OK in {bt:.1f}s, engine {plan.nbytes/1e6:.1f} MB (baked decoder engine = 124.3 MB)")

# ---- bench ----
import torch
dev = torch.device("cuda")
eng = trt.Runtime(logger).deserialize_cuda_engine(bytes(plan)); ctx = eng.create_execution_context()
def tdt(n): return {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16}[eng.get_tensor_dtype(n)]
wbuf = {n: torch.tensor(v).to(dev).to(tdt(n)) for n, v in weights.items()}
onames = [eng.get_tensor_name(i) for i in range(eng.num_io_tensors)
          if eng.get_tensor_mode(eng.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT]
def run(F):
    acts = {"asr": torch.randn(1,512,F,device=dev), "f0": torch.rand(1,2*F,device=dev)*120+80,
            "noise": torch.randn(1,2*F,device=dev), "style": torch.randn(1,128,device=dev),
            "har": torch.randn(1,1,600*F,device=dev)}
    for i in range(eng.num_io_tensors):
        n = eng.get_tensor_name(i)
        if eng.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
            t = (wbuf[n] if n in wbuf else acts[n].to(tdt(n))).contiguous()
            ctx.set_input_shape(n, tuple(t.shape)); ctx.set_tensor_address(n, t.data_ptr())
    out = torch.empty((1,1,F*300), device=dev, dtype=tdt(onames[0])); ctx.set_tensor_address(onames[0], out.data_ptr())
    ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream); return out
def timeit(F, iters=50, warm=10):
    for _ in range(warm): run(F)
    torch.cuda.synchronize(); ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        e0.record(); run(F); e1.record(); torch.cuda.synchronize(); ts.append(e0.elapsed_time(e1))
    ts.sort(); return statistics.mean(ts)
ref = {128: 6.45, 256: 10.26, 449: 19.09}
print(f"\n{'frames':>7} | {'TRT conv-w-input':>16} | baked TRT")
for F in [128, 256, 449]:
    ms = timeit(F)
    print(f"{F:>7} | {ms:14.3f}ms | ~{ref[F]}ms  -> {ms/ref[F]:.2f}x baked")
