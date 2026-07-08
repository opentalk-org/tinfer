import sys, time, statistics
sys.path.insert(0, "/tmp/ort")
import numpy as np
import onnx
from onnx import numpy_helper, TensorProto
import onnxruntime as ort

DIR = "/workspace/converted_models/libri/tensorrt"
ORT_FLOAT = {TensorProto.FLOAT, TensorProto.FLOAT16}

def promote_weights_to_inputs(src_path, dst_path):
    m = onnx.load(src_path)
    g = m.graph
    weights = {}
    keep = []
    promoted = []
    for init in g.initializer:
        if init.data_type in ORT_FLOAT:
            weights[init.name] = numpy_helper.to_array(init)
            vi = onnx.helper.make_tensor_value_info(
                init.name, init.data_type, list(init.dims))
            promoted.append(vi)
        else:
            keep.append(init)  # keep int64/int32 shape constants baked
    del g.initializer[:]
    g.initializer.extend(keep)
    g.input.extend(promoted)
    onnx.save(m, dst_path)
    tot = sum(w.nbytes for w in weights.values())
    return weights, tot, len(promoted)

def np_dtype(t):
    return {"tensor(float16)": np.float16, "tensor(float)": np.float32,
            "tensor(int64)": np.int64, "tensor(int32)": np.int32}[t]

def make_sess(path):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, so, providers=["CUDAExecutionProvider"])

def bench(sess, act_shapes, weights_gpu=None, iters=50, warm=10):
    ins = sess.get_inputs(); outs = sess.get_outputs()
    io = sess.io_binding()
    held = []
    for i in ins:
        if weights_gpu is not None and i.name in weights_gpu:
            io.bind_ortvalue_input(i.name, weights_gpu[i.name])
        else:
            arr = np.random.randn(*act_shapes[i.name]).astype(np_dtype(i.type))
            ov = ort.OrtValue.ortvalue_from_numpy(arr, "cuda", 0)
            held.append(ov); io.bind_ortvalue_input(i.name, ov)
    for o in outs: io.bind_output(o.name, "cuda", 0)
    for _ in range(warm): sess.run_with_iobinding(io)
    io.synchronize_outputs()
    ts=[]
    for _ in range(iters):
        io.synchronize_inputs(); t0=time.perf_counter()
        sess.run_with_iobinding(io); io.synchronize_outputs()
        ts.append((time.perf_counter()-t0)*1000)
    ts.sort(); return statistics.mean(ts), ts[len(ts)//2], min(ts)

def upload_weights(weights):
    return {n: ort.OrtValue.ortvalue_from_numpy(w, "cuda", 0) for n,w in weights.items()}

def run_model(name, onnx_path, shape_fn, cases):
    print(f"\n{'='*74}\n{name}\n{'='*74}")
    wpath = onnx_path.replace(".onnx", "_winput.onnx")
    weights, wbytes, ncount = promote_weights_to_inputs(onnx_path, wpath)
    print(f"promoted {ncount} weight tensors to inputs, total {wbytes/1e6:.1f} MB (hot-swap upload cost)")
    sess_norm = make_sess(onnx_path)
    sess_wi   = make_sess(wpath)
    # confirm providers
    print(f"providers norm={sess_norm.get_providers()[0]}  winput={sess_wi.get_providers()[0]}")
    wgpu = upload_weights(weights)
    print(f"{'case':>8} | {'ORT baked':>10} | {'ORT w-input':>11} | {'slowdown':>8}")
    for c in cases:
        sh = shape_fn(c)
        mn,_,_ = bench(sess_norm, sh)
        mw,_,_ = bench(sess_wi, sh, weights_gpu=wgpu)
        print(f"{str(c):>8} | {mn:8.3f}ms | {mw:9.3f}ms | {mw/mn:7.2f}x")

def dif_shapes(T):
    return {"noise":(1,1,256),"step_noise":(1,4,1,256),"embedding":(1,T,768),"features":(1,256)}
def dec_shapes(F):
    return {"asr":(1,512,F),"f0":(1,2*F),"noise":(1,2*F),"style":(1,128),"har":(1,1,600*F)}

run_model("DIFFUSION s5", f"{DIR}/diffusion_dynamic_s5.onnx", dif_shapes, [128,171,256])
run_model("DECODER", f"{DIR}/decoder_dynamic.onnx", dec_shapes, [128,256,449])
print("\n(TRT engine baseline for reference: diffusion T=171 ~1.10ms, T=256 ~1.26ms;")
print(" decoder F=128 ~6.45ms, F=256 ~10.26ms, F=449 ~19.09ms)")
