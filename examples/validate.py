import sys, json, numpy as np
sys.path.insert(0,"/tmp/ort")
import tensorrt as trt, torch
dev=torch.device("cuda"); lg=trt.Logger(trt.Logger.ERROR)
DIR="/workspace/converted_models/libri/e2e_fp16/"
meta=json.load(open(DIR+"meta.json"))
L=meta["L"]; STEPS=meta["steps"]; H=meta["harmonic_num"]+1; US=meta["upsample_scale"]
SR=meta["sampling_rate"]; THR=meta["voiced_threshold"]; SAMP=meta["sine_amp"]; NSTD=meta["noise_std"]
np.seterr(over="ignore")
def _u01(x):
    x=x.astype(np.uint32); x=(x^np.uint32(61))^(x>>np.uint32(16)); x=x*np.uint32(9); x=x^(x>>np.uint32(4)); x=x*np.uint32(0x27d4eb2d); x=x^(x>>np.uint32(15))
    return (x&np.uint32(0xFFFFFF)).astype(np.float32)*np.float32(1.0/16777216.0)
def _gauss(s):
    s=s.astype(np.uint32); u1=_u01(s*np.uint32(2)+np.uint32(1)); u2=_u01(s*np.uint32(2)+np.uint32(2)); u1=np.maximum(u1,1e-7)
    return (np.sqrt(-2*np.log(u1))*np.cos(2*np.pi*u2)).astype(np.float32)
toks=torch.tensor(meta["tokens"],dtype=torch.int64,device=dev).unsqueeze(0)
def rd(f): return np.fromfile(DIR+f,dtype=np.float32)
ref_s=torch.tensor(rd("inp_ref_s.bin").reshape(1,256),device=dev).half()
diff_noise=torch.tensor(rd("inp_diff_noise.bin").reshape(1,1,256),device=dev).half()
step_noise=torch.tensor(rd("inp_step_noise.bin").reshape(1,STEPS-1,256),device=dev).half()
rand_ini=torch.tensor(rd("inp_rand_ini.bin").reshape(1,H),device=dev)
lin=torch.tensor(rd("nsf_lin.bin"),device=dev)  # (H+1,)
F=int(open("/tmp/cpp_F.txt").read())
print("L",L,"F",F,"STEPS",STEPS,"H",H,"US",US)

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
    return run,e
runA,_=mk(DIR+"A.plan"); runB,_=mk(DIR+"B.plan"); runC,_=mk(DIR+"C.plan")

sc=lambda v: torch.tensor(v,device=dev).half()
oa=runA({"tokens":toks,"input_lengths":torch.tensor([L],dtype=torch.int64,device=dev),"ref_s":ref_s,
         "diff_noise":diff_noise,"step_noise":step_noise,"num_steps":sc(STEPS),"alpha":sc(0.7),"beta":sc(0.3)})
pd=oa["pred_dur"].float()
# alignment (match kAlign)
def build_aln(pd,F):
    cum=torch.cumsum(torch.round(pd),dim=1); start=cum-torch.round(pd)
    fidx=torch.arange(F,device=dev).view(1,1,-1)
    return ((fidx>=start.unsqueeze(-1))&(fidx<cum.unsqueeze(-1))).half()
aln=build_aln(pd,F)
ob=runB({"t_en":oa["t_en"],"d":oa["d"],"sty":oa["sty"],"aln":aln})
F0=ob["F0"].float()  # (1,2F)
twoF=F0.shape[1]
# reference har matching CUDA kPhase+kHar
rad=(F0[:,:,None]*torch.arange(1,H+1,device=dev).float())/SR; rad=rad-torch.floor(rad)  # no rand_ini (washed out)
cyc=torch.cumsum(rad,dim=1)*US  # (1,twoF,H) phase in cycles (x upsample_scale)
cyc_up=torch.nn.functional.interpolate(cyc.transpose(1,2),scale_factor=US,mode="linear",align_corners=False).transpose(1,2)
fp=(cyc_up-torch.floor(cyc_up))[0]  # (T,H) fractional phase
T=fp.shape[0]
uv=(F0[0].repeat_interleave(US)>THR).float()[:T]
tt=np.arange(T,dtype=np.int64); kk=np.arange(H,dtype=np.int64); seeds=(tt[:,None]*H+kk[None,:])
noise=torch.tensor(_gauss(seeds),device=dev)
namp=uv*NSTD+(1-uv)*(SAMP/3.0)
sine=torch.sin(2*np.pi*fp)*SAMP*uv[:,None] + namp[:,None]*noise
har=torch.tanh(sine@lin[:H]+lin[H]).view(1,1,-1).half()
oc=runC({"asr":ob["asr"],"F0":ob["F0"],"N":ob["N"],"ref":oa["ref"],"har":har})
audio=oc["audio"].float().flatten()

def cmp(name,py,cppf):
    c=torch.tensor(np.fromfile(cppf,dtype=np.float32),device=dev)
    a=py.flatten().float(); n=min(a.numel(),c.numel()); a=a[:n]; c=c[:n]
    cos=torch.dot(a,c)/(a.norm()*c.norm()+1e-9)
    print(f"  {name:9s} py{tuple(py.shape)} cpp[{c.numel()}] cos={cos.item():.5f} maxdiff={ (a-c).abs().max().item():.4f}")
cmp("pred_dur",pd,"/tmp/cpp_pred_dur.bin")
cmp("F0",F0,"/tmp/cpp_F0.bin")
cmp("har",har,"/tmp/cpp_har.bin")
cmp("audio",audio,"/tmp/cpp_audio.bin")
print("audio range py [%.3f,%.3f]"%(audio.min().item(),audio.max().item()))
# phase comparison
cp=torch.tensor(np.fromfile("/tmp/cpp_phase.bin",dtype=np.float32),device=dev)
pf=phase.flatten().float()
n=min(pf.numel(),cp.numel())
print("  phase     py[%d] cpp[%d] cos=%.5f maxdiff=%.4f  py_rng[%.2f,%.2f] cpp_rng[%.2f,%.2f]"%(
  pf.numel(),cp.numel(),(torch.dot(pf[:n],cp[:n])/(pf[:n].norm()*cp[:n].norm())).item(),(pf[:n]-cp[:n]).abs().max().item(),
  pf.min().item(),pf.max().item(),cp.min().item(),cp.max().item()))
