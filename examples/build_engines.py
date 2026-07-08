import sys, time, json, os
sys.path.insert(0, "/tmp/ort"); sys.path.insert(0, "/workspace/tinfer/tinfer")
import e2e
e2e.MAX_STEPS=5   # real 5-step diffusion (the standard diffusion_steps), no masked-10 waste
import torch, torch.nn as nn, numpy as np, onnx
from onnx import numpy_helper, helper, TensorProto, shape_inference
import tensorrt as trt
from onnxconverter_common import float16

m=e2e.m; dev=e2e.dev
sample_diffusion=e2e.sample_diffusion; build_alignment=e2e.build_alignment; length_to_mask=e2e.length_to_mask
logger=trt.Logger(trt.Logger.WARNING)
OUT="/workspace/converted_models/libri/e2e_fp16"; os.makedirs(OUT,exist_ok=True)

# proper hifigan 1-frame shift, export-safe (narrow, dynamic length)
def shift1(t):
    L=t.shape[-1]
    return torch.cat([t[..., :1], t.narrow(-1, 0, L-1)], dim=-1)

from tinfer.models.impl.styletts2.model.modules.decoder_blocks import UpSample1d as DUp, SineGen
from tinfer.models.impl.styletts2.model.modules.blocks import UpSample1d as BUp
from tinfer.models.impl.styletts2.model.modules.hifigan import Generator as HGen
def up1d(self,x): return x if self.layer_type=="none" else x.repeat_interleave(2,dim=-1)
DUp.forward=up1d; BUp.forward=up1d
# NSF: replace Resize/interpolate with repeat_interleave + manual linear-up (so C++/CUDA can replicate exactly)
def _linear_up(x, us):
    xn=torch.cat([x[:,1:,:], x[:,-1:,:]],dim=1)
    r=(torch.arange(us,device=x.device,dtype=x.dtype)/us).view(1,1,us,1)
    y=x.unsqueeze(2)*(1-r)+xn.unsqueeze(2)*r
    return y.reshape(x.shape[0],-1,x.shape[2])
def gen_pre_f0(self,f0):
    up=int(self.f0_upsamp.scale_factor); f0u=f0[:,None].repeat_interleave(up,dim=2).transpose(1,2)
    hs,_,_=self.m_source(f0u); return hs.transpose(1,2)
HGen._preprocess_f0=gen_pre_f0
def sine_f02sine(self,f0v):
    us=int(self.upsample_scale); rad=(f0v/self.sampling_rate)%1
    ri=torch.rand(f0v.shape[0],f0v.shape[2],device=f0v.device); ri=torch.cat([torch.zeros_like(ri[:,:1]),ri[:,1:]],dim=1)
    rad=torch.cat([rad[:,:1,:]+ri.unsqueeze(1),rad[:,1:,:]],dim=1)
    rad_ds=rad.reshape(rad.shape[0],-1,us,rad.shape[2])[:,:,0,:]
    phase=torch.cumsum(rad_ds,dim=1)*2*np.pi
    return torch.sin(_linear_up(phase,us))
SineGen._f02sine=sine_f02sine

class StageA(nn.Module):
    def __init__(s): super().__init__(); s.te=m.text_encoder; s.bert=m.bert; s.be=m.bert_encoder; s.pr=m.predictor
    def forward(s,tokens,il,ref_s,dn,sn,ns,alpha,beta):
        L=tokens.shape[1]; tm=length_to_mask(il,L)
        t_en=s.te(tokens,il,tm); bert_dur=s.bert(tokens,attention_mask=(~tm).int()); d_en=s.be(bert_dur).transpose(-1,-2)
        sp=sample_diffusion(dn,bert_dur,ref_s,sn,ns)[:,0]
        sty=sp[:,128:]; ref=sp[:,:128]; ref=alpha*ref+(1-alpha)*ref_s[:,:128]; sty=beta*sty+(1-beta)*ref_s[:,128:]
        d=s.pr.text_encoder(d_en,sty,il,tm); x,_=s.pr.lstm(d)
        dur=torch.sigmoid(s.pr.duration_proj(x)).sum(-1); pred_dur=torch.round(dur).clamp(min=1)*(~tm).float()
        return pred_dur,t_en,d,sty,ref
class StageB(nn.Module):
    def __init__(s): super().__init__(); s.pr=m.predictor
    def forward(s,t_en,d,sty,aln):
        en=shift1(torch.bmm(d.transpose(-1,-2),aln)); F0,N=s.pr.F0Ntrain(en,sty)
        asr=shift1(torch.bmm(t_en,aln)); return F0,N,asr
class StageC(nn.Module):
    def __init__(s): super().__init__(); s.dec=m.decoder
    def forward(s,asr,F0,N,ref,har): return s.dec.forward_with_har(asr,F0,N,ref,har)
A=StageA().eval(); Bm=StageB().eval(); C=StageC().eval()

B,L=2,171; ns=torch.tensor(5.0,device=dev)
tok=torch.randint(1,178,(B,L),device=dev); il=torch.full((B,),L,device=dev,dtype=torch.long)
rs=torch.randn(B,256,device=dev); dn=torch.randn(B,1,256,device=dev); sn=torch.randn(B,e2e.MAX_STEPS-1,256,device=dev)
with torch.no_grad():
    pd,t_en,d,sty,ref=A(tok,il,rs,dn,sn,ns,torch.tensor(0.7,device=dev),torch.tensor(0.3,device=dev))
    Fv=int(pd.sum(1).max().item()); aln=build_alignment(pd,Fv)
    F0,N,asr=Bm(t_en,d,sty,aln); har=m.decoder.generator._preprocess_f0(F0); audio=C(asr,F0,N,ref,har)
print(f"[torch] F={Fv} F0{tuple(F0.shape)} asr{tuple(asr.shape)} har{tuple(har.shape)} audio{tuple(audio.shape)} finite={torch.isfinite(audio).all().item()}",flush=True)

def export(mod,path,args,inn,outn,dyn):
    torch.onnx.export(mod,args,path,opset_version=17,input_names=inn,output_names=outn,dynamic_axes=dyn,do_constant_folding=True,dynamo=False)
FLOATOPS={"Add","Sub","Mul","Div","Pow","Clip","Min","Max","Sqrt","Reciprocal","Neg","Where","Sum","Mean","Sin","Cos","Sigmoid","Tanh","Exp","Log","Abs"}
def to_fp16(src,dst):   # bake weights fp16 + reconcile mixed-type float ops
    mo=float16.convert_float_to_float16(onnx.load(src,load_external_data=True),keep_io_types=False,disable_shape_infer=True,op_block_list=[])
    for init in mo.graph.initializer:
        if init.data_type==TensorProto.FLOAT: init.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(init).astype(np.float16),init.name))
    try: mo=shape_inference.infer_shapes(mo)
    except Exception: pass
    dt={}
    for vi in list(mo.graph.value_info)+list(mo.graph.input)+list(mo.graph.output): dt[vi.name]=vi.type.tensor_type.elem_type
    for init in mo.graph.initializer: dt[init.name]=init.data_type
    new=[]; c=0
    for node in mo.graph.node:
        if node.op_type in FLOATOPS:
            st=1 if node.op_type=="Where" else 0
            for i in range(st,len(node.input)):
                if node.input[i] and dt.get(node.input[i])==TensorProto.FLOAT:
                    co=f"{node.input[i]}__c{c}"; c+=1; new.append(helper.make_node("Cast",[node.input[i]],[co],to=TensorProto.FLOAT16)); node.input[i]=co
        new.append(node)
    del mo.graph.node[:]; mo.graph.node.extend(new); onnx.save(mo,dst)

def build(path,prof,name):
    b=trt.Builder(logger); net=b.create_network(0); p=trt.OnnxParser(net,logger)
    if not p.parse_from_file(path):
        for i in range(p.num_errors): print("PARSE",p.get_error(i));
        return False
    cfg=b.create_builder_config(); cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,8<<30)
    for fl in ("FP16","kFP16"):
        if hasattr(trt.BuilderFlag,fl): cfg.set_flag(getattr(trt.BuilderFlag,fl)); break
    pr=b.create_optimization_profile()
    for i in range(net.num_inputs):
        n=net.get_input(i).name
        if n in prof: pr.set_shape(n,*prof[n])
    cfg.add_optimization_profile(pr)
    t0=time.time(); plan=b.build_serialized_network(net,cfg)
    if plan is None: print(f"BUILD FAILED {name}"); return False
    open(f"{OUT}/{name}.plan","wb").write(bytes(plan)); print(f"  {name}: {time.time()-t0:.0f}s {plan.nbytes/1e6:.0f}MB",flush=True); return True

print("export/convert/build ...",flush=True)
export(A,"/tmp/eA.onnx",(tok,il,rs,dn,sn,ns,torch.tensor(0.7,device=dev),torch.tensor(0.3,device=dev)),
    ["tokens","input_lengths","ref_s","diff_noise","step_noise","num_steps","alpha","beta"],["pred_dur","t_en","d","sty","ref"],
    {"tokens":{0:"B",1:"L"},"input_lengths":{0:"B"},"ref_s":{0:"B"},"diff_noise":{0:"B"},"step_noise":{0:"B"},"pred_dur":{0:"B",1:"L"},"t_en":{0:"B",2:"L"},"d":{0:"B",1:"L"},"sty":{0:"B"},"ref":{0:"B"}})
export(Bm,"/tmp/eB.onnx",(t_en,d,sty,aln),["t_en","d","sty","aln"],["F0","N","asr"],
    {"t_en":{0:"B",2:"L"},"d":{0:"B",1:"L"},"sty":{0:"B"},"aln":{0:"B",1:"L",2:"F"},"F0":{0:"B",1:"F2"},"N":{0:"B",1:"F2"},"asr":{0:"B",2:"F"}})
export(C,"/tmp/eC.onnx",(asr,F0,N,ref,har),["asr","F0","N","ref","har"],["audio"],
    {"asr":{0:"B",2:"F"},"F0":{0:"B",1:"F2"},"N":{0:"B",1:"F2"},"ref":{0:"B"},"har":{0:"B",2:"H"},"audio":{0:"B",2:"T"}})
for s in ("eA","eB","eC"): to_fp16(f"/tmp/{s}.onnx",f"/tmp/{s}16.onnx")
MB=16; MT=256; MF=512; OB=8   # OB = optimization (opt) batch: TRT tunes tactics for this batch
pA={"tokens":((1,16),(OB,171),(MB,MT)),"input_lengths":((1,),(OB,),(MB,)),"ref_s":((1,256),(OB,256),(MB,256)),
    "diff_noise":((1,1,256),(OB,1,256),(MB,1,256)),"step_noise":((1,e2e.MAX_STEPS-1,256),(OB,e2e.MAX_STEPS-1,256),(MB,e2e.MAX_STEPS-1,256))}
pB={"t_en":((1,512,16),(OB,512,171),(MB,512,MT)),"d":((1,16,640),(OB,171,640),(MB,MT,640)),"sty":((1,128),(OB,128),(MB,128)),"aln":((1,16,128),(OB,171,430),(MB,MT,MF))}
pC={"asr":((1,512,128),(OB,512,430),(MB,512,MF)),"F0":((1,256),(OB,860),(MB,2*MF)),"N":((1,256),(OB,860),(MB,2*MF)),"ref":((1,128),(OB,128),(MB,128)),"har":((1,1,76800),(OB,1,258000),(MB,1,600*MF))}
ok = build("/tmp/eA16.onnx",pA,"A") and build("/tmp/eB16.onnx",pB,"B") and build("/tmp/eC16.onnx",pC,"C")
# save 150-char tokens
_pad="$";_punct=';:,.!?¡¿—…"«»“” ';_let='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_ipa="ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
SYM=[_pad]+list(_punct)+list(_let)+list(_ipa); DD={c:i for i,c in enumerate(SYM)}
TEXT=("The quick brown fox jumps over the lazy dog while the morning sun rose gently above the calm and quiet river beside the old grey stone bridge in town.")
from phonemizer.backend import EspeakBackend
ipa=EspeakBackend("en-us",preserve_punctuation=True,with_stress=True).phonemize([TEXT])[0].strip()
TOKS=[0]+[DD[c] for c in ipa if c in DD]
# NSF params for the C++/CUDA har kernel
ms=m.decoder.generator.m_source; sg=ms.l_sin_gen
lin_w=ms.l_linear.weight.detach().float().cpu().numpy().reshape(-1)  # (harmonic_num+1,)
lin_b=float(ms.l_linear.bias.detach().float().cpu().item())
np.concatenate([lin_w,[lin_b]]).astype(np.float32).tofile(f"{OUT}/nsf_lin.bin")
# fixed test inputs (seeded) for reproducible validation, batch 1
g=torch.Generator(device="cpu").manual_seed(1234)
diff_noise=torch.randn(1,1,256,generator=g); step_noise=torch.randn(1,e2e.MAX_STEPS-1,256,generator=g)
ref_s_fixed=torch.load("/workspace/converted_models/libri/voices/libri_f1.pth",map_location="cpu",weights_only=False).reshape(1,256).float()  # real voice
rand_ini=torch.rand(1,sg.harmonic_num+1,generator=g); rand_ini[:,0]=0
for nm,t in [("diff_noise",diff_noise),("step_noise",step_noise),("ref_s",ref_s_fixed),("rand_ini",rand_ini)]:
    t.numpy().astype(np.float32).tofile(f"{OUT}/inp_{nm}.bin")
json.dump({"tokens":TOKS,"L":len(TOKS),"MB":MB,"MT":MT,"MF":MF,"steps":5,
           "sampling_rate":int(sg.sampling_rate),"harmonic_num":int(sg.harmonic_num),
           "upsample_scale":int(sg.upsample_scale),"voiced_threshold":float(sg.voiced_threshold),
           "sine_amp":float(sg.sine_amp),"noise_std":float(sg.noise_std)},open(f"{OUT}/meta.json","w"))
print(f"SAVED engines+meta+nsf+inputs to {OUT}; 150 chars -> {len(TOKS)} tokens; build_ok={ok}",flush=True)
