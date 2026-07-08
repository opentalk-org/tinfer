import sys, statistics
sys.path.insert(0, "/tmp/ort")
import tensorrt as trt, torch

# ---- phonemize the real 150-char sentence -> token count ----
_pad="$"; _punct=';:,.!?¡¿—…"«»“” '
_letters='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_ipa="ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
SYM = [_pad] + list(_punct) + list(_letters) + list(_ipa)
D = {c: i for i, c in enumerate(SYM)}
TEXT = ("The quick brown fox jumps over the lazy dog while the morning sun rose "
        "gently above the calm and quiet river beside the old grey stone bridge in town.")
assert len(TEXT) == 150, len(TEXT)
from phonemizer.backend import EspeakBackend
ipa = EspeakBackend("en-us", preserve_punctuation=True, with_stress=True).phonemize([TEXT])[0].strip()
toks = [0] + [D[c] for c in ipa if c in D]
L = len(toks)
print(f"150 chars -> {L} phoneme tokens")

dev = torch.device("cuda")
logger = trt.Logger(trt.Logger.ERROR)
eng = trt.Runtime(logger).deserialize_cuda_engine(open("/tmp/e2e.plan","rb").read())
ctx = eng.create_execution_context()
TDT = {trt.DataType.FLOAT:torch.float32, trt.DataType.HALF:torch.float16, trt.DataType.INT64:torch.int64}
innames=[eng.get_tensor_name(i) for i in range(eng.num_io_tensors) if eng.get_tensor_mode(eng.get_tensor_name(i))==trt.TensorIOMode.INPUT]
outnames=[eng.get_tensor_name(i) for i in range(eng.num_io_tensors) if eng.get_tensor_mode(eng.get_tensor_name(i))==trt.TensorIOMode.OUTPUT]

NSTEPS, MAXS = 5, 10
def make():
    return {
        "tokens": torch.tensor(toks, device=dev, dtype=torch.int64).unsqueeze(0),
        "input_lengths": torch.tensor([L], device=dev, dtype=torch.int64),
        "ref_s": torch.randn(1,256,device=dev),
        "diff_noise": torch.randn(1,1,256,device=dev),
        "step_noise": torch.randn(1,MAXS-1,256,device=dev),
        "num_steps": torch.tensor(float(NSTEPS),device=dev),
        "alpha": torch.tensor(0.7,device=dev),
        "beta": torch.tensor(0.3,device=dev),
    }
def once():
    ins = make()
    for n in innames:
        t=ins[n].contiguous(); ctx.set_input_shape(n,tuple(t.shape)); ctx.set_tensor_address(n,t.data_ptr())
    outs=[]
    for n in outnames:
        sh=tuple(ctx.get_tensor_shape(n)); o=torch.empty(sh,device=dev,dtype=TDT[eng.get_tensor_dtype(n)]); outs.append(o)
        ctx.set_tensor_address(n,o.data_ptr())
    e0=torch.cuda.Event(enable_timing=True); e1=torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(); e0.record()
    ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1), outs

for _ in range(10): once()
ts = sorted(once()[0] for _ in range(50))
_, outs = once()
valid = outs[1].flatten()[0].item()
audio_s = valid*600/24000.0
ms = statistics.mean(ts)
print(f"\n=== MONOLITHIC single-engine (fp32, MAXF=512, MAX_STEPS=10 masked, num_steps=5) ===")
print(f"tokens={L}  valid_frames={valid:.0f}  audio={audio_s:.2f}s")
print(f"latency mean {ms:.2f} ms   p50 {ts[len(ts)//2]:.2f}   min {min(ts):.2f}   max {max(ts):.2f}")
print(f"RTF {ms/1000/audio_s:.4f}  ({audio_s/(ms/1000):.0f}x realtime)")
print(f"\ncompare: earlier partial-TRT (fp16 decoder+diffusion engines + torch glue): ~44 ms warm model_forward")
