import tensorrt as trt
lg = trt.Logger(trt.Logger.ERROR)
eng = trt.Runtime(lg).deserialize_cuda_engine(open("/tmp/dec_barrier.plan","rb").read())
print("num_optimization_profiles:", eng.num_optimization_profiles)
for i in range(eng.num_io_tensors):
    n = eng.get_tensor_name(i)
    if eng.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
        sh = tuple(eng.get_tensor_shape(n))
        dyn = any(d == -1 for d in sh)
        if n in ("asr","f0","noise","style","har") or dyn:
            mn,op,mx = eng.get_tensor_profile_shape(n,0)
            print(f"  {n:8s} decl={sh}  min{tuple(mn)} opt{tuple(op)} max{tuple(mx)}  {'DYNAMIC' if dyn else 'fixed'}")
