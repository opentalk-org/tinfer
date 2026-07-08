import sys, time, statistics
sys.path.insert(0, "/tmp/ort")
import tensorrt as trt, torch
logger = trt.Logger(trt.Logger.ERROR)
dev = torch.device("cuda")
eng = trt.Runtime(logger).deserialize_cuda_engine(open("/tmp/e2e.plan","rb").read())
ctx = eng.create_execution_context()
TDT = {trt.DataType.FLOAT:torch.float32, trt.DataType.HALF:torch.float16,
       trt.DataType.INT64:torch.int64, trt.DataType.INT32:torch.int32, trt.DataType.BOOL:torch.bool}
innames=[eng.get_tensor_name(i) for i in range(eng.num_io_tensors) if eng.get_tensor_mode(eng.get_tensor_name(i))==trt.TensorIOMode.INPUT]
outnames=[eng.get_tensor_name(i) for i in range(eng.num_io_tensors) if eng.get_tensor_mode(eng.get_tensor_name(i))==trt.TensorIOMode.OUTPUT]
print("inputs", innames, "outputs", outnames)

def run(B, L, NSTEPS=5, MAXS=10):
    ins = {
        "tokens": torch.randint(1,178,(B,L),device=dev,dtype=torch.int64),
        "input_lengths": torch.full((B,),L,device=dev,dtype=torch.int64),
        "ref_s": torch.randn(B,256,device=dev),
        "diff_noise": torch.randn(B,1,256,device=dev),
        "step_noise": torch.randn(B,MAXS-1,256,device=dev),
        "num_steps": torch.tensor(float(NSTEPS),device=dev),
        "alpha": torch.tensor(0.7,device=dev),
        "beta": torch.tensor(0.3,device=dev),
    }
    for n in innames:
        t=ins[n].contiguous(); ctx.set_input_shape(n,tuple(t.shape)); ctx.set_tensor_address(n,t.data_ptr())
    outs={}
    for n in outnames:
        sh=tuple(ctx.get_tensor_shape(n)); o=torch.empty(sh,device=dev,dtype=TDT[eng.get_tensor_dtype(n)]); outs[n]=o
        ctx.set_tensor_address(n,o.data_ptr())
    torch.cuda.synchronize()
    ok=ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    if not ok: raise RuntimeError("exec failed")
    return outs

for (B,L,S) in [(1,171,5),(2,120,3),(4,200,8),(1,171,10)]:
    o = run(B,L,S)
    a = o["audio"]; v = o["valid"]
    print(f"B={B} L={L} steps={S}: audio {tuple(a.shape)} valid {v.flatten().tolist()} "
          f"finite={torch.isfinite(a).all().item()} rng[{a.min().item():.3f},{a.max().item():.3f}]")
