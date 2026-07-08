import sys, time, statistics
sys.path.insert(0, "/tmp/ort"); sys.path.insert(0, "/workspace/tinfer/tinfer")
import e2e   # loads model + monkeypatches + helpers
import torch, torch.nn as nn, numpy as np, onnx
from onnx import numpy_helper, helper, TensorProto
import tensorrt as trt

m = e2e.m; dev = e2e.dev; MAX_STEPS = e2e.MAX_STEPS

# ===== replace NSF Resize/interpolate with repeat_interleave + manual linear-up (no Resize op) =====
from tinfer.models.impl.styletts2.model.modules.hifigan import Generator as HGen
from tinfer.models.impl.styletts2.model.modules.decoder_blocks import SineGen, UpSample1d

def upsample1d_forward(self, x):             # nearest x2 -> repeat_interleave (no Resize)
    if self.layer_type == "none":
        return x
    return x.repeat_interleave(2, dim=-1)
UpSample1d.forward = upsample1d_forward

def _linear_up(x, us):                       # (B,T,C) -> (B,T*us,C) linear, no Resize
    xn = torch.cat([x[:, 1:, :], x[:, -1:, :]], dim=1)
    r = (torch.arange(us, device=x.device, dtype=x.dtype) / us).view(1, 1, us, 1)
    y = x.unsqueeze(2) * (1 - r) + xn.unsqueeze(2) * r
    return y.reshape(x.shape[0], -1, x.shape[2])

def gen_preprocess_f0(self, f0):             # f0 (B, 2F) ; nearest upsample via repeat_interleave
    up = int(self.f0_upsamp.scale_factor)
    f0u = f0[:, None].repeat_interleave(up, dim=2).transpose(1, 2)   # (B, 2F*up, 1)
    har_source, _, _ = self.m_source(f0u)
    return har_source.transpose(1, 2)
HGen._preprocess_f0 = gen_preprocess_f0

def sine_f02sine(self, f0_values):           # (B, Ta, h)
    us = int(self.upsample_scale)
    rad = (f0_values / self.sampling_rate) % 1
    rand_ini = torch.rand(f0_values.shape[0], f0_values.shape[2], device=f0_values.device)
    rand_ini = torch.cat([torch.zeros_like(rand_ini[:, :1]), rand_ini[:, 1:]], dim=1)
    rad = torch.cat([rad[:, :1, :] + rand_ini.unsqueeze(1), rad[:, 1:, :]], dim=1)
    B_, Ta, h = rad.shape[0], rad.shape[1], rad.shape[2]
    rad_ds = rad.reshape(B_, -1, us, h)[:, :, 0, :]                  # nearest down to frame rate
    phase = torch.cumsum(rad_ds, dim=1) * 2 * np.pi
    return torch.sin(_linear_up(phase, us))                         # linear up, no Resize
SineGen._f02sine = sine_f02sine
sample_diffusion = e2e.sample_diffusion; build_alignment = e2e.build_alignment
shift1 = e2e.shift1; length_to_mask = e2e.length_to_mask
logger = trt.Logger(trt.Logger.ERROR)
FLOATS = {TensorProto.FLOAT, TensorProto.FLOAT16}

# ================= modules =================
class StageA(nn.Module):
    def __init__(s):
        super().__init__(); s.text_encoder=m.text_encoder; s.bert=m.bert
        s.bert_encoder=m.bert_encoder; s.predictor=m.predictor
    def forward(s, tokens, input_lengths, ref_s, diff_noise, step_noise, num_steps, alpha, beta):
        L=tokens.shape[1]; tm=length_to_mask(input_lengths, L)
        t_en=s.text_encoder(tokens, input_lengths, tm)
        bert_dur=s.bert(tokens, attention_mask=(~tm).int())
        d_en=s.bert_encoder(bert_dur).transpose(-1,-2)
        sp=sample_diffusion(diff_noise, bert_dur, ref_s, step_noise, num_steps)[:,0]
        sty=sp[:,128:]; ref=sp[:,:128]
        ref=alpha*ref+(1-alpha)*ref_s[:,:128]
        sty=beta*sty+(1-beta)*ref_s[:,128:]
        d=s.predictor.text_encoder(d_en, sty, input_lengths, tm)
        x,_=s.predictor.lstm(d)
        dur=torch.sigmoid(s.predictor.duration_proj(x)).sum(-1)
        pred_dur=torch.round(dur).clamp(min=1)*(~tm).float()
        return pred_dur, t_en, d, sty, ref

class StageB(nn.Module):
    def __init__(s):
        super().__init__(); s.predictor=m.predictor; s.decoder=m.decoder
    def forward(s, t_en, d, sty, ref, aln):
        en=shift1(torch.bmm(d.transpose(-1,-2), aln))
        F0,N=s.predictor.F0Ntrain(en, sty)
        asr=shift1(torch.bmm(t_en, aln))
        return s.decoder(asr, F0, N, ref)

A=StageA().eval(); Bm=StageB().eval()

# ================= torch validation =================
B,L=2,171; ns=torch.tensor(5.0,device=dev)
tok=torch.randint(1,178,(B,L),device=dev); il=torch.full((B,),L,device=dev,dtype=torch.long)
rs=torch.randn(B,256,device=dev); dn=torch.randn(B,1,256,device=dev); sn=torch.randn(B,MAX_STEPS-1,256,device=dev)
with torch.no_grad():
    pred_dur,t_en,d,sty,ref=A(tok,il,rs,dn,sn,ns,torch.tensor(0.7,device=dev),torch.tensor(0.3,device=dev))
    Fv=int(pred_dur.sum(1).max().item())
    aln=build_alignment(pred_dur, Fv)
    audio=Bm(t_en,d,sty,ref,aln)
print(f"[torch] pred_dur{tuple(pred_dur.shape)} t_en{tuple(t_en.shape)} d{tuple(d.shape)} aln{tuple(aln.shape)} F={Fv} audio{tuple(audio.shape)} finite={torch.isfinite(audio).all().item()}")

# ================= export helpers =================
def export(mod, path, args, input_names, output_names, dyn):
    torch.onnx.export(mod, args, path, opset_version=17, input_names=input_names,
        output_names=output_names, dynamic_axes=dyn, do_constant_folding=True, dynamo=False)

def promote(src, dst, barrier_ops=True):
    mo=onnx.load(src, load_external_data=True); g=mo.graph
    weights={}; promoted=[]
    for init in list(g.initializer):
        if init.data_type in FLOATS:
            weights[init.name]=numpy_helper.to_array(init).copy()
            promoted.append(helper.make_tensor_value_info(init.name, init.data_type, list(init.dims)))
    keep=[i for i in g.initializer if i.name not in weights]
    del g.initializer[:]; g.initializer.extend(keep); g.input.extend(promoted)
    if barrier_ops:
        BARR=set(); existing={o.name for o in g.output}
        for n in g.node:
            if n.op_type in ("ConvTranspose","Resize"): BARR.add(n.output[0])
            elif n.op_type=="Conv" and any(k in n.name for k in ("F0_conv","N_conv","conv_post")): BARR.add(n.output[0])
            elif n.op_type=="Tanh" and "decoder" in n.name: BARR.add(n.output[0])
            elif "m_source" in n.name and n.op_type in ("Add","Mul","Sin","Concat"): BARR.add(n.output[0])
        for t in BARR:
            if t not in existing: g.output.append(helper.make_tensor_value_info(t, TensorProto.FLOAT, None))
    onnx.save(mo, dst)
    wshape={vi.name:[dd.dim_value for dd in vi.type.tensor_type.shape.dim] for vi in g.input}
    return weights, wshape

def build(onnx_path, act_profiles, wshape, fp16):
    b=trt.Builder(logger); net=b.create_network(0); p=trt.OnnxParser(net, logger)
    if not p.parse_from_file(onnx_path):
        for i in range(p.num_errors): print("PARSE",p.get_error(i))
        return None
    cfg=b.create_builder_config(); cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8<<30)
    if fp16:
        for fl in ("FP16","kFP16"):
            if hasattr(trt.BuilderFlag,fl): cfg.set_flag(getattr(trt.BuilderFlag,fl)); break
    prof=b.create_optimization_profile()
    for i in range(net.num_inputs):
        n=net.get_input(i).name
        if n in act_profiles: prof.set_shape(n,*act_profiles[n])
        elif n in ("num_steps","alpha","beta"): pass
        else: s=tuple(wshape[n]); prof.set_shape(n,s,s,s)
    cfg.add_optimization_profile(prof)
    t0=time.time(); plan=b.build_serialized_network(net,cfg); dt=time.time()-t0
    if plan is None: print("BUILD FAILED"); return None
    print(f"  built in {dt:.1f}s ({plan.nbytes/1e6:.0f} MB){' fp16' if fp16 else ' fp32'}")
    return bytes(plan)

# ================= export both stages =================
print("exporting Stage A ...", flush=True)
export(A, "/tmp/stageA.onnx",
    (tok,il,rs,dn,sn,ns,torch.tensor(0.7,device=dev),torch.tensor(0.3,device=dev)),
    ["tokens","input_lengths","ref_s","diff_noise","step_noise","num_steps","alpha","beta"],
    ["pred_dur","t_en","d","sty","ref"],
    {"tokens":{0:"B",1:"L"},"input_lengths":{0:"B"},"ref_s":{0:"B"},"diff_noise":{0:"B"},"step_noise":{0:"B"},
     "pred_dur":{0:"B",1:"L"},"t_en":{0:"B",2:"L"},"d":{0:"B",1:"L"},"sty":{0:"B"},"ref":{0:"B"}})
print("exporting Stage B ...", flush=True)
export(Bm, "/tmp/stageB.onnx", (t_en,d,sty,ref,aln),
    ["t_en","d","sty","ref","aln"], ["audio"],
    {"t_en":{0:"B",2:"L"},"d":{0:"B",1:"L"},"sty":{0:"B"},"ref":{0:"B"},"aln":{0:"B",1:"L",2:"F"},"audio":{0:"B",2:"T"}})

wA,shA=promote("/tmp/stageA.onnx","/tmp/stageA_w.onnx", barrier_ops=False)
wB,shB=promote("/tmp/stageB.onnx","/tmp/stageB_w.onnx", barrier_ops=True)
print(f"Stage A weights {len(wA)}, Stage B weights {len(wB)}")

profA={"tokens":((1,16),(1,171),(4,512)),"input_lengths":((1,),(1,),(4,)),"ref_s":((1,256),(1,256),(4,256)),
       "diff_noise":((1,1,256),(1,1,256),(4,1,256)),"step_noise":((1,MAX_STEPS-1,256),(1,MAX_STEPS-1,256),(4,MAX_STEPS-1,256))}
profB={"t_en":((1,512,16),(1,512,171),(4,512,512)),"d":((1,16,640),(1,171,640),(4,512,640)),
       "ref":((1,128),(1,128),(4,128)),"sty":((1,128),(1,128),(4,128)),
       "aln":((1,16,32),(1,171,512),(4,512,2048))}

# ================= phonemize 150 chars =================
_pad="$";_punct=';:,.!?¡¿—…"«»“” ';_let='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_ipa="ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
SYM=[_pad]+list(_punct)+list(_let)+list(_ipa); DD={c:i for i,c in enumerate(SYM)}
TEXT=("The quick brown fox jumps over the lazy dog while the morning sun rose "
      "gently above the calm and quiet river beside the old grey stone bridge in town.")
from phonemizer.backend import EspeakBackend
ipa=EspeakBackend("en-us",preserve_punctuation=True,with_stress=True).phonemize([TEXT])[0].strip()
TOKS=[0]+[DD[c] for c in ipa if c in DD]; Lp=len(TOKS)
print(f"150 chars -> {Lp} tokens")
TDT={trt.DataType.FLOAT:torch.float32,trt.DataType.HALF:torch.float16,trt.DataType.INT64:torch.int64}

def runner(plan, weights):
    eng=trt.Runtime(logger).deserialize_cuda_engine(plan); ctx=eng.create_execution_context()
    ins=[eng.get_tensor_name(i) for i in range(eng.num_io_tensors) if eng.get_tensor_mode(eng.get_tensor_name(i))==trt.TensorIOMode.INPUT]
    outs=[eng.get_tensor_name(i) for i in range(eng.num_io_tensors) if eng.get_tensor_mode(eng.get_tensor_name(i))==trt.TensorIOMode.OUTPUT]
    wbuf={k:torch.tensor(v).to(dev) for k,v in weights.items()}
    def call(data):
        for n in ins:
            t=(wbuf[n] if n in wbuf else data[n]).contiguous(); ctx.set_input_shape(n,tuple(t.shape)); ctx.set_tensor_address(n,t.data_ptr())
        od={}
        for n in outs:
            sh=tuple(ctx.get_tensor_shape(n)); o=torch.empty(sh,device=dev,dtype=TDT[eng.get_tensor_dtype(n)]); od[n]=o; ctx.set_tensor_address(n,o.data_ptr())
        ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        return od
    return call, outs

def ev():
    return torch.cuda.Event(enable_timing=True)

def bench(fp16):
    tag="fp16" if fp16 else "fp32"
    print(f"\n### building {tag} ...", flush=True)
    pA=build("/tmp/stageA_w.onnx", profA, shA, fp16); callA,_=runner(pA, wA)
    pB=build("/tmp/stageB_w.onnx", profB, shB, fp16); callB,_=runner(pB, wB)
    def dataA():
        return {"tokens":torch.tensor(TOKS,device=dev,dtype=torch.int64).unsqueeze(0),
                "input_lengths":torch.tensor([Lp],device=dev,dtype=torch.int64),
                "ref_s":torch.randn(1,256,device=dev),"diff_noise":torch.randn(1,1,256,device=dev),
                "step_noise":torch.randn(1,MAX_STEPS-1,256,device=dev),"num_steps":torch.tensor(5.0,device=dev),
                "alpha":torch.tensor(0.7,device=dev),"beta":torch.tensor(0.3,device=dev)}
    tA=[]; tG=[]; tB=[]; tT=[]; Fout=0
    for it in range(30):
        torch.cuda.synchronize(); a0=ev();a1=ev();g1=ev();b1=ev()
        a0.record()
        oa=callA(dataA())
        a1.record()
        pd=oa["pred_dur"]; Fv=int(pd.sum(1).max().item()); Fv=max(Fv,32)
        alnB=build_alignment(pd, Fv)
        g1.record()
        ob=callB({"t_en":oa["t_en"],"d":oa["d"],"sty":oa["sty"],"ref":oa["ref"],"aln":alnB})
        b1.record(); torch.cuda.synchronize()
        if it>=10:
            tA.append(a0.elapsed_time(a1)); tG.append(a1.elapsed_time(g1)); tB.append(g1.elapsed_time(b1)); tT.append(a0.elapsed_time(b1))
        Fout=Fv
    aud=ob["audio"]
    me=lambda z: statistics.mean(z)
    print(f"[{tag}] 150 chars: F={Fout} frames -> {aud.shape[-1]/24000:.2f}s audio  finite={torch.isfinite(aud).all().item()}")
    print(f"[{tag}] StageA {me(tA):.2f} ms | glue {me(tG):.2f} ms | StageB {me(tB):.2f} ms | TOTAL {me(tT):.2f} ms  (RTF {me(tT)/1000/(aud.shape[-1]/24000):.4f})")

bench(False)
bench(True)
