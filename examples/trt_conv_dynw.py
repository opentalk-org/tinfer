import sys, time, statistics
sys.path.insert(0, "/tmp/ort")
import numpy as np
import tensorrt as trt
import torch

logger = trt.Logger(trt.Logger.ERROR)
dev = torch.device("cuda")
F16 = trt.DataType.HALF

def fp16_flag(cfg):
    for fl in ("FP16","kFP16"):
        if hasattr(trt.BuilderFlag, fl):
            cfg.set_flag(getattr(trt.BuilderFlag, fl)); return

def build(kind, dyn, C, K, k, L, stride=1):
    b = trt.Builder(logger); net = b.create_network(0)
    act = net.add_input("act", F16, (1, C, 1, L))
    if kind == "conv":
        kshape = (K, C, 1, k)   # KCRS
        add = net.add_convolution_nd
    else:  # deconv (transposed)
        kshape = (C, K, 1, k)   # CKRS for deconvolution
        add = net.add_deconvolution_nd
    if dyn:
        ker = net.add_input("ker", F16, kshape)
        layer = add(act, K, (1, k), trt.Weights(), trt.Weights())
        layer.set_input(1, ker)
    else:
        w = np.random.randn(*kshape).astype(np.float16)
        layer = add(act, K, (1, k), trt.Weights(w), trt.Weights())
    layer.stride_nd = (1, stride)
    out = layer.get_output(0); out.name = "out"; net.mark_output(out)
    cfg = b.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    fp16_flag(cfg)
    t0 = time.time(); plan = b.build_serialized_network(net, cfg); bt = time.time() - t0
    return plan, bt, kshape

def bench(plan, kshape, C, K, k, L, dyn, iters=50, warm=10):
    eng = trt.Runtime(logger).deserialize_cuda_engine(bytes(plan))
    ctx = eng.create_execution_context()
    act = torch.randn(1, C, 1, L, device=dev, dtype=torch.float16)
    ker = torch.randn(*kshape, device=dev, dtype=torch.float16)
    onames = [eng.get_tensor_name(i) for i in range(eng.num_io_tensors)
              if eng.get_tensor_mode(eng.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT]
    def run():
        ctx.set_tensor_address("act", act.data_ptr())
        if dyn: ctx.set_tensor_address("ker", ker.data_ptr())
        osh = tuple(ctx.get_tensor_shape(onames[0]))
        o = torch.empty(osh, device=dev, dtype=torch.float16)
        ctx.set_tensor_address(onames[0], o.data_ptr())
        ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream); return o
    for _ in range(warm): run()
    torch.cuda.synchronize(); ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        e0.record(); run(); e1.record(); torch.cuda.synchronize(); ts.append(e0.elapsed_time(e1))
    ts.sort(); return statistics.mean(ts), tuple(ctx.get_tensor_shape(onames[0]))

CASES = [
    ("conv  512x512 k3  L512",     "conv",  512, 512, 3,  512, 1),
    ("deconv 512->256 k20 s10 L432","deconv",512, 256, 20, 432, 10),
]
for label, kind, C, K, k, L, stride in CASES:
    print(f"\n### {label}")
    for dyn in (False, True):
        tag = "kernel-as-INPUT" if dyn else "constant kernel"
        plan, bt, ksh = build(kind, dyn, C, K, k, L, stride)
        if plan is None:
            print(f"  {tag:16s}: BUILD FAILED"); continue
        ms, osh = bench(plan, ksh, C, K, k, L, dyn)
        print(f"  {tag:16s}: BUILD OK ({bt:.1f}s)  latency {ms:.4f} ms  out{osh}")
