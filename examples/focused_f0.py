import sys, json, numpy as np
sys.path.insert(0,"/workspace/tinfer/tinfer"); sys.path.insert(0,"/tmp/ort")
import torch, torch.nn.functional as F
import tensorrt as trt
from tinfer.models.impl.styletts2.model.modules.load_utils import load_model
from tinfer.models.impl.styletts2.model.modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
import tinfer.models.impl.styletts2.model.modules.blocks as BLK
dev=torch.device("cuda"); lg=trt.Logger(trt.Logger.ERROR)
DIR="/workspace/converted_models/libri/e2e_fp16/"; meta=json.load(open(DIR+"meta.json"))
L=meta["L"]; STEPS=meta["steps"]
def rd(f): return np.fromfile(DIR+f,dtype=np.float32)
toks=torch.tensor(meta["tokens"],dtype=torch.long,device=dev).unsqueeze(0)
ref_s=torch.tensor(rd("inp_ref_s.bin").reshape(1,256),device=dev)
diff_noise=torch.tensor(rd("inp_diff_noise.bin").reshape(1,1,256),device=dev)
step_noise=torch.tensor(rd("inp_step_noise.bin").reshape(1,STEPS-1,256),device=dev)
m,_=load_model("/workspace/converted_models/libri/model.pth",load_style_encoder=True)
for k in m: m[k]=m[k].to(dev).eval()
for k in m:
    for p in m[k].parameters(): p.requires_grad_(False)
def L2M(lengths,Lc):
    idx=torch.arange(Lc,device=lengths.device).unsqueeze(0).expand(lengths.shape[0],-1); return torch.gt(idx+1,lengths.unsqueeze(1))
def build_alignment(pd,maxf):
    cum=torch.cumsum(pd,1); st=cum-pd; fidx=torch.arange(maxf,device=pd.device).view(1,1,-1); return ((fidx>=st.unsqueeze(-1))&(fidx<cum.unsqueeze(-1))).float()
def shift1(t): return torch.cat([t[:,:,:1],t[:,:,:-1]],dim=-1)
class FedADPM2(ADPM2Sampler):
    def __init__(s,dn,sn): super().__init__(rho=1.0); s.dn=dn; s.sn=sn
    def forward(s,noise,fn,sigmas,num_steps):
        x=sigmas[0]*s.dn[:,0]
        for i in range(num_steps-1):
            sg=sigmas[i]; sn_=sigmas[i+1]; su,sd,sm=s.get_sigmas(sg,sn_)
            d=(x-fn(x,sigma=sg))/sg; xm=x+d*(sm-sg); dm=(xm-fn(xm,sigma=sm))/sm; x=x+dm*(sd-sg); x=x+s.sn[:,i]*su
        return x
sampler=DiffusionSampler(m.diffusion.diffusion,sampler=FedADPM2(diff_noise,step_noise),sigma_schedule=KarrasSchedule(0.0001,3.0,9.0),clamp=False)
with torch.no_grad():
    il=torch.tensor([L],device=dev); tm=L2M(il,L)
    t_en=m.text_encoder(toks,il,tm); bert_dur=m.bert(toks,attention_mask=(~tm).int()); d_en=m.bert_encoder(bert_dur).transpose(-1,-2)
    s_pred=sampler(noise=diff_noise,embedding=bert_dur,features=ref_s,num_steps=STEPS).squeeze(1)
    s=s_pred[:,128:]; ref=s_pred[:,:128]; ref=0.7*ref+0.3*ref_s[:,:128]; s=0.3*s+0.7*ref_s[:,128:]
    d=m.predictor.text_encoder(d_en,s,il,tm)
    x,_=m.predictor.lstm(d); dur=torch.sigmoid(m.predictor.duration_proj(x)).sum(-1)
    pred_dur=torch.round(dur).clamp(min=1)*(~tm).float(); Fv=int(pred_dur.sum(1).max().item()); aln=build_alignment(pred_dur,Fv)
    en=shift1(torch.bmm(d.transpose(-1,-2),aln))
    F0o,No=m.predictor.F0Ntrain(en,s)                       # ORIGINAL fp32
# apply e2e monkeypatches
def adain_forward(self,x,ss):
    h=self.fc(ss).unsqueeze(-1); gamma,beta=torch.chunk(h,2,dim=1)
    mu=x.mean(2,keepdim=True); var=x.var(2,keepdim=True,unbiased=False)
    return (1+gamma)*((x-mu)/torch.sqrt(var+1e-5))+beta
BLK.AdaIN1d.forward=adain_forward
def up1d(self,x): return x if self.layer_type=="none" else x.repeat_interleave(2,dim=-1)
BLK.UpSample1d.forward=up1d
with torch.no_grad():
    F0p,Np=m.predictor.F0Ntrain(en,s)                       # PATCHED fp32
def cmp(n,a,b):
    a=a.flatten().float();b=b.flatten().float();c=torch.dot(a,b)/(a.norm()*b.norm()+1e-12);print(f"  {n}: cos={c.item():.6f} relL2={(a-b).norm()/(b.norm()+1e-12):.4f}")
print("F0Ntrain isolation (same en,s):")
cmp("F0 patched_fp32 vs original_fp32",F0p,F0o)
cmp("N  patched_fp32 vs original_fp32",Np,No)
# engine B fp16 (fed original en via t_en/d/aln)
TDT={trt.DataType.FLOAT:torch.float32,trt.DataType.HALF:torch.float16,trt.DataType.INT64:torch.int64}
e=trt.Runtime(lg).deserialize_cuda_engine(open(DIR+"B.plan","rb").read()); c=e.create_execution_context()
ins=[e.get_tensor_name(i) for i in range(e.num_io_tensors) if e.get_tensor_mode(e.get_tensor_name(i))==trt.TensorIOMode.INPUT]
outs=[e.get_tensor_name(i) for i in range(e.num_io_tensors) if e.get_tensor_mode(e.get_tensor_name(i))==trt.TensorIOMode.OUTPUT]
din={"t_en":t_en.half(),"d":d.half(),"sty":s.half(),"aln":aln.half()}; od={}
for n in ins: t=din[n].contiguous(); c.set_input_shape(n,tuple(t.shape)); c.set_tensor_address(n,t.data_ptr())
for n in outs: sh=tuple(c.get_tensor_shape(n)); o=torch.empty(sh,device=dev,dtype=TDT[e.get_tensor_dtype(n)]); od[n]=o; c.set_tensor_address(n,o.data_ptr())
c.execute_async_v3(torch.cuda.current_stream().cuda_stream); torch.cuda.synchronize()
cmp("F0 engine_fp16 (B alone)      vs original_fp32",od["F0"],F0o)
# now run engine A first, THEN engine B again -> does A corrupt B?
eA=trt.Runtime(lg).deserialize_cuda_engine(open(DIR+"A.plan","rb").read()); cA=eA.create_execution_context()
insA=[eA.get_tensor_name(i) for i in range(eA.num_io_tensors) if eA.get_tensor_mode(eA.get_tensor_name(i))==trt.TensorIOMode.INPUT]
outsA=[eA.get_tensor_name(i) for i in range(eA.num_io_tensors) if eA.get_tensor_mode(eA.get_tensor_name(i))==trt.TensorIOMode.OUTPUT]
dinA={"tokens":toks,"input_lengths":torch.tensor([L],dtype=torch.long,device=dev),"ref_s":ref_s.half(),"diff_noise":diff_noise.half(),
      "step_noise":step_noise.half(),"num_steps":torch.tensor(STEPS,device=dev).half(),"alpha":torch.tensor(0.7,device=dev).half(),"beta":torch.tensor(0.3,device=dev).half()}
odA={}
for n in insA: t=dinA[n].contiguous(); cA.set_input_shape(n,tuple(t.shape)); cA.set_tensor_address(n,t.data_ptr())
for n in outsA: sh=tuple(cA.get_tensor_shape(n)); o=torch.empty(sh,device=dev,dtype=TDT[eA.get_tensor_dtype(n)]); odA[n]=o; cA.set_tensor_address(n,o.data_ptr())
cA.execute_async_v3(torch.cuda.current_stream().cuda_stream); torch.cuda.synchronize()
# re-run B after A
od2={}
for n in ins: t=din[n].contiguous(); c.set_input_shape(n,tuple(t.shape)); c.set_tensor_address(n,t.data_ptr())
for n in outs: sh=tuple(c.get_tensor_shape(n)); o=torch.empty(sh,device=dev,dtype=TDT[e.get_tensor_dtype(n)]); od2[n]=o; c.set_tensor_address(n,o.data_ptr())
c.execute_async_v3(torch.cuda.current_stream().cuda_stream); torch.cuda.synchronize()
cmp("F0 engine_fp16 (B after A)    vs original_fp32",od2["F0"],F0o)
