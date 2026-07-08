import sys, json, numpy as np
sys.path.insert(0,"/workspace/tinfer/tinfer"); sys.path.insert(0,"/tmp/ort")
import torch, torch.nn.functional as F
import tensorrt as trt
from tinfer.models.impl.styletts2.model.modules.load_utils import load_model
from tinfer.models.impl.styletts2.model.modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
dev=torch.device("cuda"); lg=trt.Logger(trt.Logger.ERROR)
DIR="/workspace/converted_models/libri/e2e_fp16/"; meta=json.load(open(DIR+"meta.json"))
L=meta["L"]; STEPS=meta["steps"]; US=meta["upsample_scale"]; SR=meta["sampling_rate"]; THR=meta["voiced_threshold"]; SAMP=meta["sine_amp"]
def rd(f): return np.fromfile(DIR+f,dtype=np.float32)
toks=torch.tensor(meta["tokens"],dtype=torch.long,device=dev).unsqueeze(0)
ref_s=torch.tensor(rd("inp_ref_s.bin").reshape(1,256),device=dev)
diff_noise=torch.tensor(rd("inp_diff_noise.bin").reshape(1,1,256),device=dev)
step_noise=torch.tensor(rd("inp_step_noise.bin").reshape(1,STEPS-1,256),device=dev)
rand_ini=torch.tensor(rd("inp_rand_ini.bin").reshape(1,-1),device=dev)

m,_=load_model("/workspace/converted_models/libri/model.pth",load_style_encoder=True)
for k in m: m[k]=m[k].to(dev).eval()
for k in m:
    for p in m[k].parameters(): p.requires_grad_(False)

def L2M(lengths,Lc):
    idx=torch.arange(Lc,device=lengths.device).unsqueeze(0).expand(lengths.shape[0],-1); return torch.gt(idx+1,lengths.unsqueeze(1))
def build_alignment(pd,maxf):
    cum=torch.cumsum(pd,1); st=cum-pd; fidx=torch.arange(maxf,device=pd.device).view(1,1,-1)
    return ((fidx>=st.unsqueeze(-1))&(fidx<cum.unsqueeze(-1))).float()
def shift1(t): return torch.cat([t[:,:,:1],t[:,:,:-1]],dim=-1)
class FedADPM2(ADPM2Sampler):
    def __init__(s,dn,sn): super().__init__(rho=1.0); s.dn=dn; s.sn=sn
    def forward(s,noise,fn,sigmas,num_steps):
        x=sigmas[0]*s.dn[:,0]
        for i in range(num_steps-1):
            sg=sigmas[i]; sn_=sigmas[i+1]; su,sd,sm=s.get_sigmas(sg,sn_)
            d=(x-fn(x,sigma=sg))/sg; xm=x+d*(sm-sg); dm=(xm-fn(xm,sigma=sm))/sm; x=x+dm*(sd-sg); x=x+s.sn[:,i]*su
        return x
sampler=DiffusionSampler(m.diffusion.diffusion,sampler=FedADPM2(diff_noise,step_noise),
    sigma_schedule=KarrasSchedule(sigma_min=0.0001,sigma_max=3.0,rho=9.0),clamp=False)

# ---- original intermediates ----
O={}
with torch.no_grad():
    il=torch.tensor([L],device=dev); tm=L2M(il,L)
    O["t_en"]=m.text_encoder(toks,il,tm)
    bert_dur=m.bert(toks,attention_mask=(~tm).int()); d_en=m.bert_encoder(bert_dur).transpose(-1,-2)
    s_pred=sampler(noise=diff_noise,embedding=bert_dur,features=ref_s,num_steps=STEPS).squeeze(1)
    O["s_pred"]=s_pred
    s=s_pred[:,128:]; ref=s_pred[:,:128]; ref=0.7*ref+0.3*ref_s[:,:128]; s=0.3*s+0.7*ref_s[:,128:]
    O["s"]=s; O["ref"]=ref
    O["d"]=m.predictor.text_encoder(d_en,s,il,tm)
    x,_=m.predictor.lstm(O["d"]); dur=torch.sigmoid(m.predictor.duration_proj(x)).sum(-1)
    O["pred_dur"]=torch.round(dur).clamp(min=1)*(~tm).float()
    Fv=int(O["pred_dur"].sum(1).max().item()); aln=build_alignment(O["pred_dur"],Fv)
    en=shift1(torch.bmm(O["d"].transpose(-1,-2),aln)); F0,N=m.predictor.F0Ntrain(en,s)
    O["F0"]=F0; O["N"]=N; O["asr"]=shift1(torch.bmm(O["t_en"],aln))
    O["har"]=m.decoder.generator._preprocess_f0(F0)   # ORIGINAL NSF

# ---- TRT engines ----
TDT={trt.DataType.FLOAT:torch.float32,trt.DataType.HALF:torch.float16,trt.DataType.INT64:torch.int64}
def mk(path):
    e=trt.Runtime(lg).deserialize_cuda_engine(open(path,"rb").read()); c=e.create_execution_context()
    ins=[e.get_tensor_name(i) for i in range(e.num_io_tensors) if e.get_tensor_mode(e.get_tensor_name(i))==trt.TensorIOMode.INPUT]
    outs=[e.get_tensor_name(i) for i in range(e.num_io_tensors) if e.get_tensor_mode(e.get_tensor_name(i))==trt.TensorIOMode.OUTPUT]
    def run(d):
        for n in ins: t=d[n].contiguous(); c.set_input_shape(n,tuple(t.shape)); c.set_tensor_address(n,t.data_ptr())
        od={}
        for n in outs: sh=tuple(c.get_tensor_shape(n)); o=torch.empty(sh,device=dev,dtype=TDT[e.get_tensor_dtype(n)]); od[n]=o; c.set_tensor_address(n,o.data_ptr())
        c.execute_async_v3(torch.cuda.current_stream().cuda_stream); torch.cuda.synchronize(); return od
    return run
runA=mk(DIR+"A.plan"); runB=mk(DIR+"B.plan")
h=lambda t:t.half()
oa=runA({"tokens":toks,"input_lengths":torch.tensor([L],dtype=torch.long,device=dev),"ref_s":h(ref_s),
    "diff_noise":h(diff_noise),"step_noise":h(step_noise),
    "num_steps":torch.tensor(STEPS,device=dev).half(),"alpha":torch.tensor(0.7,device=dev).half(),"beta":torch.tensor(0.3,device=dev).half()})
# stage B fed with ORIGINAL inputs (isolate B): use original t_en,d,s + aln(original pred_dur)
aln=build_alignment(O["pred_dur"],int(O["pred_dur"].sum(1).max().item()))
ob=runB({"t_en":h(O["t_en"]),"d":h(O["d"]),"sty":h(O["s"]),"aln":h(aln)})

def cmp(name,a,b):
    a=a.flatten().float(); b=b.flatten().float(); n=min(a.numel(),b.numel()); a=a[:n]; b=b[:n]
    cos=torch.dot(a,b)/(a.norm()*b.norm()+1e-12)
    rel=(a-b).norm()/(b.norm()+1e-12)
    print(f"  {name:9s} cos={cos.item():.6f} relL2={rel.item():.4f}  orig[{b.min():+.3f},{b.max():+.3f}] trt[{a.min():+.3f},{a.max():+.3f}]")
print("STAGE-BY-STAGE (TRT engine fp16  vs  original model fp32):")
cmp("t_en",oa["t_en"],O["t_en"])
cmp("s(sty)",oa["sty"],O["s"])
cmp("ref",oa["ref"],O["ref"])
cmp("d",oa["d"],O["d"])
cmp("pred_dur",oa["pred_dur"],O["pred_dur"])
print("  F0 shapes: engineB",tuple(ob["F0"].shape),"orig",tuple(O["F0"].shape)); cmp("F0",ob["F0"],O["F0"])
cmp("N",ob["N"],O["N"])
cmp("asr",ob["asr"],O["asr"])
# C++ runner F0 (full A->B->C pipeline)
import os
if os.path.exists("/tmp/cpp_F0.bin"):
    cppF0=torch.tensor(np.fromfile("/tmp/cpp_F0.bin",dtype=np.float32),device=dev)
    cmp("F0 C++runner",cppF0,O["F0"])
