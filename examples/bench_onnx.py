import sys, time, statistics
sys.path.insert(0, "/tmp/ort")
import numpy as np
import onnxruntime as ort

print("ORT", ort.__version__, "providers:", ort.get_available_providers())
DIR = "/workspace/converted_models/libri/tensorrt"

def make_session(path):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, sess_options=so, providers=["CUDAExecutionProvider"])

def np_dtype(t):
    return {"tensor(float16)": np.float16, "tensor(float)": np.float32,
            "tensor(int64)": np.int64, "tensor(int32)": np.int32}[t]

def bench(path, shape_fn, cases, label):
    sess = make_session(path)
    print(f"\n=== {label}  provider={sess.get_providers()[0]} ===")
    ins = sess.get_inputs()
    outs = sess.get_outputs()
    for case in cases:
        shapes = shape_fn(case)
        feed_np = {i.name: np.random.randn(*shapes[i.name]).astype(np_dtype(i.type)) for i in ins}
        # upload inputs to GPU once
        io = sess.io_binding()
        gpu_inputs = {}
        for i in ins:
            ov = ort.OrtValue.ortvalue_from_numpy(feed_np[i.name], "cuda", 0)
            gpu_inputs[i.name] = ov
            io.bind_ortvalue_input(i.name, ov)
        for o in outs:
            io.bind_output(o.name, "cuda", 0)  # keep output on GPU
        # warm
        for _ in range(10):
            sess.run_with_iobinding(io)
        io.synchronize_outputs()
        ts = []
        for _ in range(50):
            io.synchronize_inputs()
            t0 = time.perf_counter()
            sess.run_with_iobinding(io)
            io.synchronize_outputs()
            ts.append((time.perf_counter() - t0) * 1000)
        ts.sort()
        print(f"  {case}: mean {statistics.mean(ts):7.3f} ms  p50 {ts[len(ts)//2]:7.3f}  min {min(ts):7.3f}")

# diffusion
def dif_shapes(T):
    return {"noise": (1,1,256), "step_noise": (1,4,1,256), "embedding": (1,T,768), "features": (1,256)}
bench(f"{DIR}/diffusion_dynamic_s5.onnx", dif_shapes, [64,128,171,256], "DIFFUSION s5 (ONNX/ORT-CUDA)")

# decoder
def dec_shapes(F):
    return {"asr": (1,512,F), "f0": (1,2*F), "noise": (1,2*F), "style": (1,128), "har": (1,1,600*F)}
bench(f"{DIR}/decoder_dynamic.onnx", dec_shapes, [128,256,449,512], "DECODER (ONNX/ORT-CUDA)")
