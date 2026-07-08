import sys, json, numpy as np
sys.path.insert(0,"/workspace/tinfer/tinfer"); sys.path.insert(0,"/tmp/ort")
import torch, torch.nn.functional as F
from tinfer.models.impl.styletts2.model.modules.load_utils import load_model
from tinfer.models.impl.styletts2.model.modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
dev=torch.device("cuda")
DIR="/workspace/converted_models/libri/e2e_fp16/"
meta=json.load(open(DIR+"meta.json")); L=meta["L"]; STEPS=meta["steps"]
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

np.seterr(over="ignore")
def _u01(x):
    x=x.astype(np.uint32); x=(x^np.uint32(61))^(x>>np.uint32(16)); x=x*np.uint32(9); x=x^(x>>np.uint32(4)); x=x*np.uint32(0x27d4eb2d); x=x^(x>>np.uint32(15))
    return (x&np.uint32(0xFFFFFF)).astype(np.float32)*np.float32(1.0/16777216.0)
def _gauss(s):
    s=s.astype(np.uint32); u1=_u01(s*np.uint32(2)+np.uint32(1)); u2=_u01(s*np.uint32(2)+np.uint32(2)); u1=np.maximum(u1,1e-7)
    return (np.sqrt(-2*np.log(u1))*np.cos(2*np.pi*u2)).astype(np.float32)
def length_to_mask(lengths,Lc):
    idx=torch.arange(Lc,device=lengths.device).unsqueeze(0).expand(lengths.shape[0],-1)
    return torch.gt(idx+1,lengths.unsqueeze(1))
def build_alignment(pred_dur,maxf):
    cum=torch.cumsum(pred_dur,dim=1); start=cum-pred_dur
    fidx=torch.arange(maxf,device=pred_dur.device).view(1,1,-1)
    return ((fidx>=start.unsqueeze(-1))&(fidx<cum.unsqueeze(-1))).float()
def shift1(t): return torch.cat([t[:,:,:1],t[:,:,:-1]],dim=-1)

# original ADPM2 sampler but with fed noise (deterministic)
class FedADPM2(ADPM2Sampler):
    def __init__(self,dn,sn): super().__init__(rho=1.0); self.dn=dn; self.sn=sn
    def forward(self,noise,fn,sigmas,num_steps):
        x=sigmas[0]*self.dn[:,0]
        for i in range(num_steps-1):
            sigma=sigmas[i]; sigma_next=sigmas[i+1]
            su,sd,sm=self.get_sigmas(sigma,sigma_next)
            d=(x-fn(x,sigma=sigma))/sigma; x_mid=x+d*(sm-sigma)
            d_mid=(x_mid-fn(x_mid,sigma=sm))/sm; x=x+d_mid*(sd-sigma)
            x=x+self.sn[:,i]*su
        return x
sampler=DiffusionSampler(m.diffusion.diffusion,sampler=FedADPM2(diff_noise,step_noise),
    sigma_schedule=KarrasSchedule(sigma_min=0.0001,sigma_max=3.0,rho=9.0),clamp=False)

def original_forward(fill_nsf):
    tm=length_to_mask(torch.tensor([L],device=dev),L)
    il=torch.tensor([L],device=dev)
    t_en=m.text_encoder(toks,il,tm)
    bert_dur=m.bert(toks,attention_mask=(~tm).int())
    d_en=m.bert_encoder(bert_dur).transpose(-1,-2)
    s_pred=sampler(noise=diff_noise,embedding=bert_dur,features=ref_s,num_steps=STEPS).squeeze(1)
    s=s_pred[:,128:]; ref=s_pred[:,:128]
    ref=0.7*ref+0.3*ref_s[:,:128]; s=0.3*s+0.7*ref_s[:,128:]
    d=m.predictor.text_encoder(d_en,s,il,tm)
    x,_=m.predictor.lstm(d)
    dur=torch.sigmoid(m.predictor.duration_proj(x)).sum(-1)
    pred_dur=torch.round(dur).clamp(min=1)*(~tm).float()
    Fv=int(pred_dur.sum(1).max().item()); aln=build_alignment(pred_dur,Fv)
    en=shift1(torch.bmm(d.transpose(-1,-2),aln)); F0,N=m.predictor.F0Ntrain(en,s)
    asr=shift1(torch.bmm(t_en,aln))
    if fill_nsf: _patch_nsf_fixed()
    else: _unpatch_nsf()
    audio=m.decoder(asr,F0,N,ref)
    return audio.squeeze().float().cpu().numpy(), Fv, pred_dur.sum(1).item(), s_pred

# NSF patch to match CUDA har (fixed rand_ini, zero additive noise, repeat_interleave+manual linear-up)
_orig_f02sine=m.decoder.generator.m_source.l_sin_gen._f02sine
_orig_sinegen_fwd=m.decoder.generator.m_source.l_sin_gen.forward
_orig_pre=m.decoder.generator._preprocess_f0
def _patch_nsf_fixed():
    sg=m.decoder.generator.m_source.l_sin_gen; gen=m.decoder.generator; ms=m.decoder.generator.m_source
    def f02sine(f0v):   # ORIGINAL math, only rand_ini fixed (deterministic)
        us=int(sg.upsample_scale); rad=(f0v/sg.sampling_rate)%1
        ri=torch.cat([torch.zeros_like(rand_ini[:,:1]),rand_ini[:,1:]],dim=1)
        rad=torch.cat([rad[:,:1,:]+ri.unsqueeze(1),rad[:,1:,:]],dim=1)
        rad=F.interpolate(rad.transpose(1,2),scale_factor=1/us,mode="linear").transpose(1,2)
        phase=torch.cumsum(rad,dim=1)*2*np.pi
        phase=F.interpolate(phase.transpose(1,2)*us,scale_factor=us,mode="linear").transpose(1,2)
        return torch.sin(phase)
    def sine_fwd(f0):
        fn=f0*torch.arange(1,sg.harmonic_num+2,device=f0.device)
        sines=f02sine(fn)*sg.sine_amp; uv=(f0>sg.voiced_threshold).float()
        Bn,Tn,Hn=sines.shape
        bb=np.arange(Bn)[:,None,None]; tt=np.arange(Tn)[None,:,None]; kk=np.arange(Hn)[None,None,:]
        seeds=((bb*Tn+tt)*Hn+kk).astype(np.int64)
        noise=torch.tensor(_gauss(seeds),device=f0.device)
        namp=uv*sg.noise_std+(1-uv)*sg.sine_amp/3
        return sines*uv+namp*noise, uv, noise
    def pre(f0):
        up=int(gen.f0_upsamp.scale_factor); f0u=f0[:,None].repeat_interleave(up,dim=2).transpose(1,2)
        hs,_,_=ms(f0u); return hs.transpose(1,2)
    sg._f02sine=f02sine; sg.forward=sine_fwd; gen._preprocess_f0=pre
def _unpatch_nsf():
    sg=m.decoder.generator.m_source.l_sin_gen
    sg._f02sine=_orig_f02sine; sg.forward=_orig_sinegen_fwd; m.decoder.generator._preprocess_f0=_orig_pre

with torch.no_grad():
    au_faith,Fv,dsum,sp=original_forward(fill_nsf=True)
    au_real,_,_,_=original_forward(fill_nsf=False)
trt=np.fromfile("/tmp/cpp_audio.bin",dtype=np.float32)
print(f"orig F={Fv} dur_sum={dsum:.0f}  s_pred_norm={sp.norm().item():.3f}")
print(f"lengths: orig_faithful={len(au_faith)} orig_real={len(au_real)} trt={len(trt)}")

def stats(name,a):
    print(f"  {name:14s} len={len(a):7d} mean={a.mean():+.5f} std={a.std():.5f} rms={np.sqrt((a**2).mean()):.5f} min={a.min():+.4f} max={a.max():+.4f} absmean={np.abs(a).mean():.5f}")
stats("orig_faithful",au_faith); stats("orig_real",au_real); stats("trt",trt)

def windowed_rms(a,w=1200):
    n=len(a)//w; a=a[:n*w].reshape(n,w); return np.sqrt((a**2).mean(1))
def compare(name,a,b):
    n=min(len(a),len(b)); a=a[:n]; b=b[:n]
    ra,rb=windowed_rms(a),windowed_rms(b); k=min(len(ra),len(rb)); ra,rb=ra[:k],rb[:k]
    rms_corr=np.corrcoef(ra,rb)[0,1]
    rms_relerr=np.abs(ra-rb).mean()/(rb.mean()+1e-9)
    wav_corr=np.corrcoef(a,b)[0,1]
    print(f"  {name}: windowed-RMS corr={rms_corr:.4f} relerr={rms_relerr:.3f}  waveform corr={wav_corr:.4f}")
print("COMPARE (trt vs original):")
compare("trt vs orig_faithful",trt,au_faith)
compare("trt vs orig_real    ",trt,au_real)
compare("orig_real vs faithful",au_real,au_faith)
# spectrogram (magnitude) correlation - perceptually relevant, phase-invariant
def logmag(a,n=1024,hop=256):
    t=torch.tensor(a,device=dev); w=torch.hann_window(n,device=dev)
    S=torch.stft(t,n_fft=n,hop_length=hop,window=w,return_complex=True).abs()
    return torch.log(S+1e-5)
def speccmp(name,a,b):
    A=logmag(a).flatten(); B=logmag(b).flatten(); n=min(A.numel(),B.numel()); A=A[:n];B=B[:n]
    print(f"  {name}: log-mel-spec corr={torch.corrcoef(torch.stack([A,B]))[0,1].item():.4f}")
print("SPECTROGRAM (phase-invariant):")
speccmp("trt vs orig_real    ",trt,au_real)
speccmp("orig_real vs faithful",au_real,au_faith)
# har direct
cpph=np.fromfile("/tmp/cpp_har.bin",dtype=np.float32)
_patch_nsf_fixed()
with torch.no_grad(): harf=m.decoder.generator._preprocess_f0(torch.tensor(np.fromfile("/tmp/cpp_F0.bin",dtype=np.float32).reshape(1,-1),device=dev)).squeeze().float().cpu().numpy()
n=min(len(cpph),len(harf))
print(f"  har cpp vs faithful: corr={np.corrcoef(cpph[:n],harf[:n])[0,1]:.4f}  cpp_std={cpph.std():.5f} faith_std={harf.std():.5f}")
