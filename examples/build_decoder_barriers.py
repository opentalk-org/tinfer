import sys, time, statistics
sys.path.insert(0, "/tmp/ort")
import numpy as np
import onnx
from onnx import numpy_helper, helper, TensorProto
import tensorrt as trt

DIR = "/workspace/converted_models/libri/tensorrt"
SRC = f"{DIR}/decoder_dynamic_winput.onnx"
DST = f"{DIR}/decoder_dynamic_barriers.onnx"
logger = trt.Logger(trt.Logger.ERROR)
FLOATS = {TensorProto.FLOAT, TensorProto.FLOAT16}

orig = onnx.load(f"{DIR}/decoder_dynamic.onnx")
weights_np = {i.name: numpy_helper.to_array(i) for i in orig.graph.initializer if i.data_type in FLOATS}

m = onnx.load(SRC); g = m.graph
existing_out = {o.name for o in g.output}
# barrier tensors: source convs, all (de)conv-transpose upsamplers, conv_post
BARRIERS = ["/F0_conv/Conv_output_0", "/N_conv/Conv_output_0",
            "/decode.3/pool/ConvTranspose_output_0",
            "/ups.0/ConvTranspose_output_0", "/ups.1/ConvTranspose_output_0",
            "/ups.2/ConvTranspose_output_0", "/ups.3/ConvTranspose_output_0",
            "/conv_post/Conv_output_0"]
for t in BARRIERS:
    if t not in existing_out:
        g.output.append(helper.make_tensor_value_info(t, TensorProto.FLOAT16, None))
onnx.save(m, DST)
print(f"added {len(BARRIERS)} fusion-barrier outputs")

ACT = {"asr","f0","noise","style","har"}
wshapes = {vi.name: [d.dim_value for d in vi.type.tensor_type.shape.dim] for vi in g.input}
builder = trt.Builder(logger); network = builder.create_network(0)
parser = trt.OnnxParser(network, logger)
with open(DST,"rb") as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors): print("parse err", parser.get_error(i));
        sys.exit(2)
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 6<<30)
def act_shape(n,F): return {"asr":(1,512,F),"f0":(1,2*F),"noise":(1,2*F),"style":(1,128),"har":(1,1,600*F)}[n]
prof = builder.create_optimization_profile()
for i in range(network.num_inputs):
    n = network.get_input(i).name
    if n in ACT: prof.set_shape(n, act_shape(n,128), act_shape(n,256), act_shape(n,512))
    else: s=tuple(wshapes[n]); prof.set_shape(n,s,s,s)
config.add_optimization_profile(prof)
t0=time.time(); plan=builder.build_serialized_network(network,config); bt=time.time()-t0
if plan is None:
    print(f"BUILD FAILED after {bt:.1f}s"); sys.exit(1)
print(f"BUILD OK in {bt:.1f}s, engine {plan.nbytes/1e6:.1f} MB")
with open("/tmp/dec_barrier.plan","wb") as f: f.write(bytes(plan))
print("plan saved")

import torch
dev=torch.device("cuda")
eng=trt.Runtime(logger).deserialize_cuda_engine(bytes(plan)); ctx=eng.create_execution_context()
def tdt(n): return {trt.DataType.FLOAT:torch.float32,trt.DataType.HALF:torch.float16}[eng.get_tensor_dtype(n)]
wbuf={n:torch.tensor(v).to(dev).to(tdt(n)) for n,v in weights_np.items()}
io_out=[eng.get_tensor_name(i) for i in range(eng.num_io_tensors) if eng.get_tensor_mode(eng.get_tensor_name(i))==trt.TensorIOMode.OUTPUT]
def run(F):
    acts={"asr":torch.randn(1,512,F,device=dev),"f0":torch.rand(1,2*F,device=dev)*120+80,
          "noise":torch.randn(1,2*F,device=dev),"style":torch.randn(1,128,device=dev),
          "har":torch.randn(1,1,600*F,device=dev)}
    for i in range(eng.num_io_tensors):
        n=eng.get_tensor_name(i)
        if eng.get_tensor_mode(n)==trt.TensorIOMode.INPUT:
            t=(wbuf[n] if n in wbuf else acts[n].to(tdt(n))).contiguous()
            ctx.set_input_shape(n,tuple(t.shape)); ctx.set_tensor_address(n,t.data_ptr())
    keep=[]
    for n in io_out:
        sh=tuple(ctx.get_tensor_shape(n)); o=torch.empty(sh,device=dev,dtype=tdt(n)); keep.append(o)
        ctx.set_tensor_address(n,o.data_ptr())
    ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream); return keep
def timeit(F,iters=50,warm=10):
    for _ in range(warm): run(F)
    torch.cuda.synchronize(); ts=[]
    for _ in range(iters):
        torch.cuda.synchronize(); e0=torch.cuda.Event(enable_timing=True); e1=torch.cuda.Event(enable_timing=True)
        e0.record(); run(F); e1.record(); torch.cuda.synchronize(); ts.append(e0.elapsed_time(e1))
    ts.sort(); return statistics.mean(ts)
ref={128:6.45,256:10.26,449:19.09}
print(f"\n{'frames':>6} | {'conv-w-input':>12} | baked TRT")
for F in [128,256,449]:
    ms=timeit(F); print(f"{F:>6} | {ms:10.3f}ms | ~{ref[F]}ms -> {ms/ref[F]:.2f}x")
