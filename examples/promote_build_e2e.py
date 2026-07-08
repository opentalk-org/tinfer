import sys, time, statistics
sys.path.insert(0, "/tmp/ort")
import numpy as np, onnx
from onnx import numpy_helper, helper, TensorProto
import tensorrt as trt, torch

SRC = "/tmp/e2e.onnx"
DST = "/tmp/e2e_winput.onnx"
FLOATS = {TensorProto.FLOAT, TensorProto.FLOAT16}
logger = trt.Logger(trt.Logger.WARNING)
dev = torch.device("cuda")
MAXS = 10

# ---- promote ALL float weights to inputs ----
m = onnx.load(SRC, load_external_data=True)
g = m.graph
weights = {}
promoted = []
for init in list(g.initializer):
    if init.data_type in FLOATS:
        weights[init.name] = numpy_helper.to_array(init).copy()
        promoted.append(helper.make_tensor_value_info(init.name, init.data_type, list(init.dims)))
keep = [i for i in g.initializer if i.name not in weights]
del g.initializer[:]; g.initializer.extend(keep)
g.input.extend(promoted)

# ---- fusion barriers (decoder region megafusion) ----
BARR = set()
for n in g.node:
    if n.op_type == "ConvTranspose":
        BARR.add(n.output[0])
    elif n.op_type == "Conv" and any(k in n.name for k in ("F0_conv","N_conv","conv_post")):
        BARR.add(n.output[0])
    elif n.op_type == "Tanh" and "decoder" in n.name:
        BARR.add(n.output[0])
existing = {o.name for o in g.output}
for t in BARR:
    if t not in existing:
        g.output.append(helper.make_tensor_value_info(t, TensorProto.FLOAT, None))
onnx.save(m, DST)
wbytes = sum(w.nbytes for w in weights.values())
print(f"promoted {len(weights)} weights ({wbytes/1e6:.1f} MB) to inputs; barriers={len(BARR)}")

# ---- build TRT ----
builder = trt.Builder(logger); network = builder.create_network(0)
parser = trt.OnnxParser(network, logger)
if not parser.parse_from_file(DST):
    for i in range(parser.num_errors): print("PARSE ERR", parser.get_error(i))
    sys.exit(2)
ACT = {"tokens","input_lengths","ref_s","diff_noise","step_noise","num_steps","alpha","beta"}
wshape = {vi.name: [d.dim_value for d in vi.type.tensor_type.shape.dim] for vi in g.input}
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
prof = builder.create_optimization_profile()
def sp(n, mn, op, mx): prof.set_shape(n, mn, op, mx)
for i in range(network.num_inputs):
    n = network.get_input(i).name
    if n == "tokens": sp(n,(1,16),(1,171),(4,512))
    elif n == "input_lengths": sp(n,(1,),(1,),(4,))
    elif n == "ref_s": sp(n,(1,256),(1,256),(4,256))
    elif n == "diff_noise": sp(n,(1,1,256),(1,1,256),(4,1,256))
    elif n == "step_noise": sp(n,(1,MAXS-1,256),(1,MAXS-1,256),(4,MAXS-1,256))
    elif n in ("num_steps","alpha","beta"): pass  # scalars
    else:
        s = tuple(wshape[n]); sp(n, s, s, s)     # weights fixed
config.add_optimization_profile(prof)
print("building (all-weights-input, fp32) ...", flush=True)
t0=time.time(); plan=builder.build_serialized_network(network, config); bt=time.time()-t0
if plan is None:
    print(f"BUILD FAILED after {bt:.1f}s"); sys.exit(1)
print(f"BUILD OK in {bt:.1f}s, engine {plan.nbytes/1e6:.1f} MB")
open("/tmp/e2e_winput.plan","wb").write(bytes(plan))

# ---- run + bench 150 chars ----
_pad="$"; _punct=';:,.!?¡¿—…"«»“” '
_letters='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_ipa="ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
SYM=[_pad]+list(_punct)+list(_letters)+list(_ipa); D={c:i for i,c in enumerate(SYM)}
TEXT=("The quick brown fox jumps over the lazy dog while the morning sun rose "
      "gently above the calm and quiet river beside the old grey stone bridge in town.")
from phonemizer.backend import EspeakBackend
ipa=EspeakBackend("en-us",preserve_punctuation=True,with_stress=True).phonemize([TEXT])[0].strip()
toks=[0]+[D[c] for c in ipa if c in D]; L=len(toks)
print(f"150 chars -> {L} tokens")

eng=trt.Runtime(logger).deserialize_cuda_engine(bytes(plan)); ctx=eng.create_execution_context()
TDT={trt.DataType.FLOAT:torch.float32,trt.DataType.HALF:torch.float16,trt.DataType.INT64:torch.int64}
wbuf={k: torch.tensor(v).to(dev) for k,v in weights.items()}
innames=[eng.get_tensor_name(i) for i in range(eng.num_io_tensors) if eng.get_tensor_mode(eng.get_tensor_name(i))==trt.TensorIOMode.INPUT]
outnames=[eng.get_tensor_name(i) for i in range(eng.num_io_tensors) if eng.get_tensor_mode(eng.get_tensor_name(i))==trt.TensorIOMode.OUTPUT]
NSTEPS=5
def data_inputs():
    return {"tokens":torch.tensor(toks,device=dev,dtype=torch.int64).unsqueeze(0),
            "input_lengths":torch.tensor([L],device=dev,dtype=torch.int64),
            "ref_s":torch.randn(1,256,device=dev),"diff_noise":torch.randn(1,1,256,device=dev),
            "step_noise":torch.randn(1,MAXS-1,256,device=dev),"num_steps":torch.tensor(float(NSTEPS),device=dev),
            "alpha":torch.tensor(0.7,device=dev),"beta":torch.tensor(0.3,device=dev)}
def once():
    di=data_inputs()
    for n in innames:
        t=(wbuf[n] if n in wbuf else di[n]).contiguous(); ctx.set_input_shape(n,tuple(t.shape)); ctx.set_tensor_address(n,t.data_ptr())
    outs=[]
    for n in outnames:
        sh=tuple(ctx.get_tensor_shape(n)); o=torch.empty(sh,device=dev,dtype=TDT[eng.get_tensor_dtype(n)]); outs.append(o)
        ctx.set_tensor_address(n,o.data_ptr())
    e0=torch.cuda.Event(enable_timing=True); e1=torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(); e0.record(); ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream); e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1), outs
for _ in range(10): once()
ts=sorted(once()[0] for _ in range(50)); _,outs=once()
vi=outnames.index("valid"); valid=outs[vi].flatten()[0].item(); ai=outnames.index("audio"); a=outs[ai]
ms=statistics.mean(ts); audio_s=valid*600/24000
print(f"\n=== MONOLITH ALL-WEIGHTS-AS-INPUT (fp32) 150 chars ===")
print(f"data inputs=8  weight inputs={len(weights)}  finite={torch.isfinite(a).all().item()}")
print(f"valid={valid:.0f} audio={audio_s:.2f}s  latency mean {ms:.2f} ms p50 {ts[len(ts)//2]:.2f} min {min(ts):.2f}")
print(f"RTF {ms/1000/audio_s:.4f} ({audio_s/(ms/1000):.0f}x realtime)")
