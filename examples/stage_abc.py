import sys, time, statistics
sys.path.insert(0, "/tmp/ort"); sys.path.insert(0, "/workspace/tinfer/tinfer")
import e2e
import torch, torch.nn as nn, numpy as np, onnx
from onnx import numpy_helper, helper, TensorProto
import tensorrt as trt

m=e2e.m; dev=e2e.dev; MAX_STEPS=e2e.MAX_STEPS
sample_diffusion=e2e.sample_diffusion; build_alignment=e2e.build_alignment
length_to_mask=e2e.length_to_mask
shift1=lambda t: t   # hifigan 1-frame shift disabled for bench (baked-slice export bug); timing unaffected
logger=trt.Logger(trt.Logger.ERROR); FLOATS={TensorProto.FLOAT,TensorProto.FLOAT16}

# NSF/UpSample Resize -> repeat_interleave (host har uses these too; harmless on host)
from tinfer.models.impl.styletts2.model.modules.decoder_blocks import UpSample1d as DUp
from tinfer.models.impl.styletts2.model.modules.blocks import UpSample1d as BUp
def up1d(self,x):
    return x if self.layer_type=="none" else x.repeat_interleave(2,dim=-1)
DUp.forward=up1d; BUp.forward=up1d

# ---------- modules ----------
class StageA(nn.Module):
    def __init__(s): super().__init__(); s.text_encoder=m.text_encoder; s.bert=m.bert; s.bert_encoder=m.bert_encoder; s.predictor=m.predictor
    def forward(s,tokens,il,ref_s,dn,sn,ns,alpha,beta):
        L=tokens.shape[1]; tm=length_to_mask(il,L)
        t_en=s.text_encoder(tokens,il,tm); bert_dur=s.bert(tokens,attention_mask=(~tm).int())
        d_en=s.bert_encoder(bert_dur).transpose(-1,-2)
        sp=sample_diffusion(dn,bert_dur,ref_s,sn,ns)[:,0]
        sty=sp[:,128:]; ref=sp[:,:128]; ref=alpha*ref+(1-alpha)*ref_s[:,:128]; sty=beta*sty+(1-beta)*ref_s[:,128:]
        d=s.predictor.text_encoder(d_en,sty,il,tm); x,_=s.predictor.lstm(d)
        dur=torch.sigmoid(s.predictor.duration_proj(x)).sum(-1); pred_dur=torch.round(dur).clamp(min=1)*(~tm).float()
        return pred_dur,t_en,d,sty,ref

class StageB(nn.Module):
    def __init__(s): super().__init__(); s.predictor=m.predictor
    def forward(s,t_en,d,sty,aln):
        en=shift1(torch.bmm(d.transpose(-1,-2),aln)); F0,N=s.predictor.F0Ntrain(en,sty)
        asr=shift1(torch.bmm(t_en,aln)); return F0,N,asr

class StageC(nn.Module):
    def __init__(s): super().__init__(); s.decoder=m.decoder
    def forward(s,asr,F0,N,ref,har): return s.decoder.forward_with_har(asr,F0,N,ref,har)

A=StageA().eval(); Bm=StageB().eval(); C=StageC().eval()

# ---------- torch validation ----------
B,L=2,171; ns=torch.tensor(5.0,device=dev)
tok=torch.randint(1,178,(B,L),device=dev); il=torch.full((B,),L,device=dev,dtype=torch.long)
rs=torch.randn(B,256,device=dev); dn=torch.randn(B,1,256,device=dev); sn=torch.randn(B,MAX_STEPS-1,256,device=dev)
with torch.no_grad():
    pd,t_en,d,sty,ref=A(tok,il,rs,dn,sn,ns,torch.tensor(0.7,device=dev),torch.tensor(0.3,device=dev))
    Fv=int(pd.sum(1).max().item()); aln=build_alignment(pd,Fv)
    F0,N,asr=Bm(t_en,d,sty,aln)
    har=m.decoder.generator._preprocess_f0(F0)
    audio=C(asr,F0,N,ref,har)
print(f"[torch] F={Fv} F0{tuple(F0.shape)} asr{tuple(asr.shape)} har{tuple(har.shape)} audio{tuple(audio.shape)} finite={torch.isfinite(audio).all().item()}")

# ---------- helpers ----------
def export(mod,path,args,inn,outn,dyn):
    torch.onnx.export(mod,args,path,opset_version=17,input_names=inn,output_names=outn,dynamic_axes=dyn,do_constant_folding=True,dynamo=False)
def promote(src,dst,barr):
    mo=onnx.load(src,load_external_data=True); g=mo.graph; w={}; pr=[]
    for init in list(g.initializer):
        if init.data_type in FLOATS: w[init.name]=numpy_helper.to_array(init).copy(); pr.append(helper.make_tensor_value_info(init.name,init.data_type,list(init.dims)))
    keep=[i for i in g.initializer if i.name not in w]; del g.initializer[:]; g.initializer.extend(keep); g.input.extend(pr)
    if barr:
        BB=set(); ex={o.name for o in g.output}
        for n in g.node:
            if n.op_type=="ConvTranspose": BB.add(n.output[0])
            elif n.op_type=="Conv" and any(k in n.name for k in ("conv_post",)): BB.add(n.output[0])
            elif n.op_type=="Tanh" and "generator" in n.name: BB.add(n.output[0])
        for t in BB:
            if t not in ex: g.output.append(helper.make_tensor_value_info(t,TensorProto.FLOAT,None))
    onnx.save(mo,dst); wsh={vi.name:[dd.dim_value for dd in vi.type.tensor_type.shape.dim] for vi in g.input}; return w,wsh
def build(path,prof,wsh,fp16):
    b=trt.Builder(logger); net=b.create_network(0); p=trt.OnnxParser(net,logger)
    if not p.parse_from_file(path):
        for i in range(p.num_errors): print("PARSE",p.get_error(i))
        return None
    cfg=b.create_builder_config(); cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,8<<30)
    if fp16:
        for fl in ("FP16","kFP16"):
            if hasattr(trt.BuilderFlag,fl): cfg.set_flag(getattr(trt.BuilderFlag,fl)); break
    pr=b.create_optimization_profile()
    for i in range(net.num_inputs):
        n=net.get_input(i).name
        if n in prof: pr.set_shape(n,*prof[n])
        elif n in ("num_steps","alpha","beta"): pass
        else: s=tuple(wsh[n]); pr.set_shape(n,s,s,s)
    cfg.add_optimization_profile(pr)
    t0=time.time(); plan=b.build_serialized_network(net,cfg)
    if plan is None: print("BUILD FAILED"); return None
    print(f"  built {time.time()-t0:.0f}s {plan.nbytes/1e6:.0f}MB{' fp16' if fp16 else ''}")
    return bytes(plan)
TDT={trt.DataType.FLOAT:torch.float32,trt.DataType.HALF:torch.float16,trt.DataType.INT64:torch.int64}
def runner(plan,w):
    eng=trt.Runtime(logger).deserialize_cuda_engine(plan); ctx=eng.create_execution_context()
    ins=[eng.get_tensor_name(i) for i in range(eng.num_io_tensors) if eng.get_tensor_mode(eng.get_tensor_name(i))==trt.TensorIOMode.INPUT]
    outs=[eng.get_tensor_name(i) for i in range(eng.num_io_tensors) if eng.get_tensor_mode(eng.get_tensor_name(i))==trt.TensorIOMode.OUTPUT]
    wb={k:torch.tensor(v).to(dev) for k,v in w.items()}
    def call(data):
        keep=[]
        for n in ins:
            src=wb[n] if n in wb else data[n]
            t=src.to(TDT[eng.get_tensor_dtype(n)]).contiguous(); keep.append(t)
            ctx.set_input_shape(n,tuple(t.shape)); ctx.set_tensor_address(n,t.data_ptr())
        od={"_keep":keep}
        for n in outs:
            sh=tuple(ctx.get_tensor_shape(n)); o=torch.empty(sh,device=dev,dtype=TDT[eng.get_tensor_dtype(n)]); od[n]=o; ctx.set_tensor_address(n,o.data_ptr())
        ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream); return od
    return call

# ---------- export + promote ----------
print("exporting A/B/C ...",flush=True)
export(A,"/tmp/A.onnx",(tok,il,rs,dn,sn,ns,torch.tensor(0.7,device=dev),torch.tensor(0.3,device=dev)),
    ["tokens","input_lengths","ref_s","diff_noise","step_noise","num_steps","alpha","beta"],["pred_dur","t_en","d","sty","ref"],
    {"tokens":{0:"B",1:"L"},"input_lengths":{0:"B"},"ref_s":{0:"B"},"diff_noise":{0:"B"},"step_noise":{0:"B"},"pred_dur":{0:"B",1:"L"},"t_en":{0:"B",2:"L"},"d":{0:"B",1:"L"},"sty":{0:"B"},"ref":{0:"B"}})
export(Bm,"/tmp/B.onnx",(t_en,d,sty,aln),["t_en","d","sty","aln"],["F0","N","asr"],
    {"t_en":{0:"B",2:"L"},"d":{0:"B",1:"L"},"sty":{0:"B"},"aln":{0:"B",1:"L",2:"F"},"F0":{0:"B",1:"F2"},"N":{0:"B",1:"F2"},"asr":{0:"B",2:"F"}})
export(C,"/tmp/C.onnx",(asr,F0,N,ref,har),["asr","F0","N","ref","har"],["audio"],
    {"asr":{0:"B",2:"F"},"F0":{0:"B",1:"F2"},"N":{0:"B",1:"F2"},"ref":{0:"B"},"har":{0:"B",2:"H"},"audio":{0:"B",2:"T"}})
wA,shA=promote("/tmp/A.onnx","/tmp/Aw.onnx",False)
wB,shB=promote("/tmp/B.onnx","/tmp/Bw.onnx",False)
wC,shC=promote("/tmp/C.onnx","/tmp/Cw.onnx",True)
print(f"weights A={len(wA)} B={len(wB)} C={len(wC)}")
from onnxconverter_common import float16
from onnx import shape_inference
FLOATOPS={"Add","Sub","Mul","Div","Pow","Clip","Min","Max","Sqrt","Reciprocal","Neg","Where","Sum","Mean","Sin","Cos","Sigmoid","Tanh","Exp","Log","Abs"}
def to_fp16(src,dst):   # full fp16, then cast every fp32 input of a float op -> fp16 (leave int shape-ops alone)
    mo=float16.convert_float_to_float16(onnx.load(src),keep_io_types=False,disable_shape_infer=True,op_block_list=[])
    for init in mo.graph.initializer:
        if init.data_type==TensorProto.FLOAT:
            init.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(init).astype(np.float16),init.name))
    try: mo=shape_inference.infer_shapes(mo)
    except Exception: pass
    dt={}
    for vi in list(mo.graph.value_info)+list(mo.graph.input)+list(mo.graph.output): dt[vi.name]=vi.type.tensor_type.elem_type
    for init in mo.graph.initializer: dt[init.name]=init.data_type
    new=[]; c=0
    for node in mo.graph.node:
        if node.op_type in FLOATOPS:
            start=1 if node.op_type=="Where" else 0
            for i in range(start,len(node.input)):
                inp=node.input[i]
                if inp and dt.get(inp)==TensorProto.FLOAT:
                    co=f"{inp}__c{c}"; c+=1
                    new.append(helper.make_node("Cast",[inp],[co],to=TensorProto.FLOAT16)); node.input[i]=co
        new.append(node)
    del mo.graph.node[:]; mo.graph.node.extend(new)
    onnx.save(mo,dst)
for s in ("Aw","Bw","Cw"): to_fp16(f"/tmp/{s}.onnx",f"/tmp/{s}16.onnx")
print("fp16 onnx converted")
for k in ["text_encoder","bert","bert_encoder","predictor"]:  # free VRAM; keep decoder.generator for host har
    m[k].to("cpu")
torch.cuda.empty_cache()

MB=4   # max batch; frames capped at 1024 (=25.6s) so 3 contexts fit comfortably in 32GB
profA={"tokens":((1,16),(1,171),(MB,512)),"input_lengths":((1,),(1,),(MB,)),"ref_s":((1,256),(1,256),(MB,256)),
       "diff_noise":((1,1,256),(1,1,256),(MB,1,256)),"step_noise":((1,MAX_STEPS-1,256),(1,MAX_STEPS-1,256),(MB,MAX_STEPS-1,256))}
profB={"t_en":((1,512,16),(1,512,171),(MB,512,512)),"d":((1,16,640),(1,171,640),(MB,512,640)),"sty":((1,128),(1,128),(MB,128)),
       "aln":((1,16,128),(1,171,512),(MB,512,1024))}
profC={"asr":((1,512,128),(1,512,512),(MB,512,1024)),"F0":((1,256),(1,1024),(MB,2048)),"N":((1,256),(1,1024),(MB,2048)),
       "ref":((1,128),(1,128),(MB,128)),"har":((1,1,76800),(1,1,307200),(MB,1,614400))}

# ---------- phonemize ----------
_pad="$";_punct=';:,.!?¡¿—…"«»“” ';_let='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_ipa="ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
SYM=[_pad]+list(_punct)+list(_let)+list(_ipa); DD={c:i for i,c in enumerate(SYM)}
TEXT=("The quick brown fox jumps over the lazy dog while the morning sun rose gently above the calm and quiet river beside the old grey stone bridge in town.")
from phonemizer.backend import EspeakBackend
ipa=EspeakBackend("en-us",preserve_punctuation=True,with_stress=True).phonemize([TEXT])[0].strip()
TOKS=[0]+[DD[c] for c in ipa if c in DD]; Lp=len(TOKS); print(f"150 chars -> {Lp} tokens")

_v=torch.load("/workspace/converted_models/libri/voices/libri_f1.pth",map_location=dev,weights_only=False)
if isinstance(_v,dict): _v=_v.get("ref_s", next(iter(_v.values())))
REF_S=torch.as_tensor(_v,device=dev).reshape(1,256).float()   # real voice -> realistic durations
def dataA():
    return {"tokens":torch.tensor(TOKS,device=dev,dtype=torch.int64).unsqueeze(0),"input_lengths":torch.tensor([Lp],device=dev,dtype=torch.int64),
            "ref_s":REF_S,"diff_noise":torch.randn(1,1,256,device=dev),"step_noise":torch.randn(1,MAX_STEPS-1,256,device=dev),
            "num_steps":torch.tensor(5.0,device=dev),"alpha":torch.tensor(0.7,device=dev),"beta":torch.tensor(0.3,device=dev)}
ev=lambda: torch.cuda.Event(enable_timing=True)

def bench(fp16):
    tag="fp16" if fp16 else "fp32"; sfx="16" if fp16 else ""; print(f"\n### {tag}",flush=True)
    ca=runner(build(f"/tmp/Aw{sfx}.onnx",profA,shA,fp16),wA)
    cb=runner(build(f"/tmp/Bw{sfx}.onnx",profB,shB,fp16),wB)
    cc=runner(build(f"/tmp/Cw{sfx}.onnx",profC,shC,fp16),wC)
    TA=[];TG1=[];TB=[];TG2=[];TC=[];TT=[];Fo=0
    for it in range(30):
        torch.cuda.synchronize(); e0=ev();e1=ev();e2=ev();e3=ev();e4=ev();e5=ev()
        e0.record(); oa=ca(dataA()); e1.record()
        pd=oa["pred_dur"]; Fv=430  # fixed realistic frame count for 150 chars (timing is F-driven, not value-driven)
        alnB=build_alignment(pd,Fv); e2.record()
        if it==0: print("DBG pd",tuple(pd.shape),"t_en",tuple(oa["t_en"].shape),"alnB",tuple(alnB.shape),flush=True)
        ob=cb({"t_en":oa["t_en"],"d":oa["d"],"sty":oa["sty"],"aln":alnB}); e3.record()
        if it==0: print("DBG ob asr",tuple(ob["asr"].shape),"F0",tuple(ob["F0"].shape),flush=True)
        har=m.decoder.generator._preprocess_f0(ob["F0"].float()); e4.record()
        oc=cc({"asr":ob["asr"],"F0":ob["F0"],"N":ob["N"],"ref":oa["ref"],"har":har}); e5.record()
        torch.cuda.synchronize()
        if it>=10:
            TA.append(e0.elapsed_time(e1));TG1.append(e1.elapsed_time(e2));TB.append(e2.elapsed_time(e3));TG2.append(e3.elapsed_time(e4));TC.append(e4.elapsed_time(e5));TT.append(e0.elapsed_time(e5))
        Fo=Fv
    au=oc["audio"]; me=statistics.mean
    print(f"[{tag}] F={Fo} audio={au.shape[-1]/24000:.2f}s finite={torch.isfinite(au).all().item()}")
    print(f"[{tag}] A {me(TA):.2f} | align {me(TG1):.2f} | B(F0N) {me(TB):.2f} | har {me(TG2):.2f} | C(dec) {me(TC):.2f} | TOTAL {me(TT):.2f} ms  RTF {me(TT)/1000/(au.shape[-1]/24000):.4f}")

bench(False); bench(True)
print("\ncompare: earlier partial-TRT (decoder+diffusion fp16 engines + torch glue) warm model_forward ~44 ms")
