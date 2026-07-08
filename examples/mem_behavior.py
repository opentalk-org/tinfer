import sys, subprocess, os
sys.path.insert(0, "/tmp/ort")
sys.path.insert(0, "/workspace/tinfer/tinfer")
import numpy as np

def gpu_used():
    out = subprocess.check_output(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"]).decode()
    return int(out.strip().split("\n")[0])

DIR = "/workspace/converted_models/libri/tensorrt"

# ---------- TRT: fixed scratch sized to MAX profile ----------
import tensorrt as trt
logger = trt.Logger(trt.Logger.ERROR)
with open(f"{DIR}/decoder_dynamic.engine","rb") as f:
    eng = trt.Runtime(logger).deserialize_cuda_engine(f.read())
print("=== TensorRT decoder engine ===")
print(f" engine (weights) size on disk : {os.path.getsize(f'{DIR}/decoder_dynamic.engine')/1e6:.1f} MB")
print(f" device_memory_size (scratch)  : {eng.device_memory_size_v2/1e6:.1f} MB  <-- reserved for MAX profile, independent of actual batch")
# show profile max
ctx = eng.create_execution_context()
for i in range(eng.num_io_tensors):
    n = eng.get_tensor_name(i)
    if eng.get_tensor_mode(n)==trt.TensorIOMode.INPUT:
        mn,opt,mx = eng.get_tensor_profile_shape(n,0)
        print(f"   {n}: max profile {tuple(mx)}")
        break
del ctx, eng

# ---------- ORT: lazy arena, grows with ACTUAL shapes ----------
import onnxruntime as ort
print("\n=== ONNX Runtime (CUDA) decoder ===")
base = gpu_used(); print(f" gpu_used baseline                 : {base} MB")
so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess = ort.InferenceSession(f"{DIR}/decoder_dynamic.onnx", so, providers=["CUDAExecutionProvider"])
after_load = gpu_used(); print(f" after session create (weights)    : {after_load} MB  (+{after_load-base})")

def run(B,F):
    feed = {"asr":np.random.randn(B,512,F).astype(np.float16),
            "f0":np.random.randn(B,2*F).astype(np.float16),
            "noise":np.random.randn(B,2*F).astype(np.float16),
            "style":np.random.randn(B,128).astype(np.float16),
            "har":np.random.randn(B,1,600*F).astype(np.float16)}
    sess.run(None, feed)

run(1,128); m1=gpu_used(); print(f" after run batch=1  F=128           : {m1} MB  (+{m1-after_load} activations)")
run(1,128); m1b=gpu_used(); print(f" after run batch=1  F=128 (again)   : {m1b} MB  (no growth = arena cached)")
run(4,512); m4=gpu_used(); print(f" after run batch=4  F=512 (bigger)  : {m4} MB  (+{m4-m1b} arena GREW to fit)")
run(1,128); m1c=gpu_used(); print(f" back to batch=1  F=128             : {m1c} MB  (stays high = arena keeps high-water mark)")
print("\n-> ORT reserves the LARGEST SHAPE ACTUALLY RUN (lazy high-water), not the biggest possible.")
