import sys
sys.path.insert(0, "/tmp/ort")
sys.path.insert(0, "/workspace/tinfer/tinfer")
import numpy as np, onnx
from onnx import numpy_helper, TensorProto
import tensorrt as trt, torch
from tinfer.models.impl.styletts2.model.modules import tensorrt_runtime as rt

DIR = "/workspace/converted_models/libri/tensorrt"
logger = trt.Logger(trt.Logger.ERROR); dev = torch.device("cuda")
FLOATS = {TensorProto.FLOAT, TensorProto.FLOAT16}
orig = onnx.load(f"{DIR}/decoder_dynamic.onnx")
weights_np = {i.name: numpy_helper.to_array(i) for i in orig.graph.initializer if i.data_type in FLOATS}

F = 256
# identical random activation inputs
torch.manual_seed(0)
acts = {"asr":torch.randn(1,512,F,device=dev,dtype=torch.float16),
        "f0":(torch.rand(1,2*F,device=dev,dtype=torch.float16)*120+80),
        "noise":torch.randn(1,2*F,device=dev,dtype=torch.float16),
        "style":torch.randn(1,128,device=dev,dtype=torch.float16),
        "har":torch.randn(1,1,600*F,device=dev,dtype=torch.float16)}

# ---- baked engine (shared runner) ----
baked = rt.get_tensorrt_decoder_runner(DIR)
out_baked = baked.run({k:v for k,v in acts.items()})["audio"].float()

# ---- barrier engine (weights as input) ----
eng = trt.Runtime(logger).deserialize_cuda_engine(open("/tmp/dec_barrier.plan","rb").read())
ctx = eng.create_execution_context()
def tdt(n): return {trt.DataType.FLOAT:torch.float32,trt.DataType.HALF:torch.float16}[eng.get_tensor_dtype(n)]
wbuf = {n:torch.tensor(v).to(dev).to(tdt(n)) for n,v in weights_np.items()}
innames=[eng.get_tensor_name(i) for i in range(eng.num_io_tensors) if eng.get_tensor_mode(eng.get_tensor_name(i))==trt.TensorIOMode.INPUT]
outnames=[eng.get_tensor_name(i) for i in range(eng.num_io_tensors) if eng.get_tensor_mode(eng.get_tensor_name(i))==trt.TensorIOMode.OUTPUT]
for n in innames:
    t=(wbuf[n] if n in wbuf else acts[n]).contiguous()
    ctx.set_input_shape(n,tuple(t.shape)); ctx.set_tensor_address(n,t.data_ptr())
outs={}
for n in outnames:
    sh=tuple(ctx.get_tensor_shape(n)); o=torch.empty(sh,device=dev,dtype=tdt(n)); outs[n]=o
    ctx.set_tensor_address(n,o.data_ptr())
torch.cuda.synchronize(); ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream); torch.cuda.synchronize()
out_win = outs["audio"].float()

a = out_baked.flatten(); b = out_win.flatten()
n = min(a.numel(), b.numel()); a=a[:n]; b=b[:n]
print(f"baked shape {tuple(out_baked.shape)}  win shape {tuple(out_win.shape)}")
print(f"max|baked| {a.abs().max():.4f}  max|win| {b.abs().max():.4f}")
print(f"max abs diff : {(a-b).abs().max().item():.6f}")
print(f"mean abs diff: {(a-b).abs().mean().item():.6f}")
den = (a.norm()*b.norm()).item()
print(f"cosine sim   : {torch.dot(a,b).item()/den if den>0 else float('nan'):.6f}")
