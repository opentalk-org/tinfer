// StyleTTS2 end-to-end TensorRT fp16 runner. 3 engines (A prosody, B F0N, C decoder)
// + CUDA kernels for alignment and NSF harmonic source. Full model, no skipped parts.
#include "NvInferRuntime.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <chrono>
using namespace nvinfer1;

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)
class Logger : public ILogger { void log(Severity s,const char* m) noexcept override { if(s<=Severity::kERROR) fprintf(stderr,"[TRT] %s\n",m);} } gLogger;
static std::vector<char> readfile(const std::string& p){ std::ifstream f(p,std::ios::binary|std::ios::ate); if(!f){fprintf(stderr,"open %s\n",p.c_str());exit(1);} size_t n=f.tellg(); f.seekg(0); std::vector<char> b(n); f.read(b.data(),n); return b;}
static std::vector<float> readf32(const std::string& p){ auto b=readfile(p); std::vector<float> v(b.size()/4); memcpy(v.data(),b.data(),b.size()); return v;}

// ---- kernels ----
__global__ void kFillH(half* p,size_t n,float v){ size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<n)p[i]=__float2half(v);}
__global__ void kCastF2H(const float* s,half* d,size_t n){ size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<n)d[i]=__float2half(s[i]);}
__global__ void kAlign(const half* dur,half* aln,int B,int L,int F){
  int l=blockIdx.x*blockDim.x+threadIdx.x,b=blockIdx.y; if(l>=L||b>=B)return;
  int st=0; for(int j=0;j<l;j++) st+=(int)(__half2float(dur[b*L+j])+0.5f);
  int w=(int)(__half2float(dur[b*L+l])+0.5f); size_t base=((size_t)b*L+l)*F;
  for(int f=st;f<st+w&&f<F;f++) aln[base+f]=__float2half(1.0f);
}
// pass 1: rad per (b,j,k) at frame rate, fully parallel. rad = frac(f0*(k+1)/sr), +rand_ini at frame 0.
__global__ void kRad(const half* F0,const float* ri,double* ph,int B,int twoF,int H,float sr){
  size_t idx=(size_t)blockIdx.x*blockDim.x+threadIdx.x; size_t tot=(size_t)B*twoF*H; if(idx>=tot)return;
  int k=idx%H; int j=(idx/H)%twoF; int b=idx/((size_t)twoF*H);
  double f0=(double)__half2float(F0[(size_t)b*twoF+j]); double rad=f0*(k+1)/(double)sr; rad-=floor(rad);
  (void)ri; ph[idx]=rad;   // rand_ini is washed out by the original's 1/us linear downsample -> omit
}
// pass 2: in-place cumsum over j per (b,k), scaled by upsample_scale -> phase in CYCLES (matches original phase*upsample_scale)
__global__ void kScan(double* ph,int B,int twoF,int H,int us){
  int k=blockIdx.x*blockDim.x+threadIdx.x,b=blockIdx.y; if(k>=H||b>=B)return;
  double acc=0.0;
  for(int j=0;j<twoF;j++){ size_t idx=((size_t)b*twoF+j)*H+k; acc+=ph[idx]; ph[idx]=acc*(double)us;}
}
__device__ __forceinline__ float u01(unsigned x){ x=(x^61u)^(x>>16); x*=9u; x^=x>>4; x*=0x27d4eb2du; x^=x>>15; return (x&0xFFFFFFu)*(1.0f/16777216.0f); }
__device__ __forceinline__ float gauss(unsigned s){ float u1=u01(s*2u+1u),u2=u01(s*2u+2u); u1=fmaxf(u1,1e-7f); return sqrtf(-2.f*logf(u1))*cospif(2.f*u2); }
// phase (cycles) is upsampled to audio rate via F.interpolate(linear, align_corners=False), then sin(2*pi*frac(phase))
__global__ void kHar(const half* F0,const double* ph,const float* lin,half* har,int B,int twoF,int H,int us,float thr,float samp,float nstd){
  size_t T=(size_t)twoF*us; size_t t=(size_t)blockIdx.x*blockDim.x+threadIdx.x; int b=blockIdx.y; if(t>=T||b>=B)return;
  int juv=t/us; float f0u=__half2float(F0[b*twoF+juv]); float uv=(f0u>thr)?1.f:0.f;
  double x=((double)t+0.5)/us-0.5; double xf=floor(x); int j0=(int)xf; double fr=x-xf;   // align_corners=False
  int ja=j0<0?0:(j0>=twoF?twoF-1:j0); int jb=(j0+1)<0?0:((j0+1)>=twoF?twoF-1:(j0+1));
  float acc=0.f; float namp=uv*nstd+(1.f-uv)*(samp*0.33333333f);
  for(int k=0;k<H;k++){ double ca=ph[((size_t)b*twoF+ja)*H+k],cb=ph[((size_t)b*twoF+jb)*H+k];
    double loc=ca*(1.0-fr)+cb*fr; float fp=(float)(loc-floor(loc));
    unsigned seed=(unsigned)(((size_t)b*T+t)*H+k);
    float sw=sinpif(2.f*fp)*samp*uv + namp*gauss(seed);
    acc+=sw*lin[k]; }
  acc+=lin[H]; har[(size_t)b*T+t]=__float2half(tanhf(acc));
}

struct Eng{ ICudaEngine* eng=nullptr; IExecutionContext* ctx=nullptr; std::vector<std::string> ins,outs;
  void load(IRuntime* rt,const std::string& p){ auto b=readfile(p); eng=rt->deserializeCudaEngine(b.data(),b.size());
    if(!eng){fprintf(stderr,"deser %s\n",p.c_str());exit(1);} ctx=eng->createExecutionContext();
    for(int i=0;i<eng->getNbIOTensors();i++){const char* n=eng->getIOTensorName(i);
      (eng->getTensorIOMode(n)==TensorIOMode::kINPUT?ins:outs).push_back(n);} }
};
static Dims D(std::initializer_list<int> v){ Dims d; d.nbDims=v.size(); int i=0; for(int x:v)d.d[i++]=x; return d;}
static size_t vol(const Dims& d){ size_t v=1; for(int i=0;i<d.nbDims;i++)v*=d.d[i]; return v;}

// global buffers (max sized)
std::map<std::string,void*> buf;   // name -> gpu ptr
void* alloc(const std::string& n,size_t bytes){ void* p; CK(cudaMalloc(&p,bytes)); buf[n]=p; return p;}
cudaStream_t stream;

int main(int argc,char** argv){
  const std::string DIR="/workspace/converted_models/libri/e2e_fp16/";
  auto meta=readfile(DIR+"meta.json"); std::string ms(meta.begin(),meta.end());
  auto ji=[&](const char* k){auto p=ms.find(std::string("\"")+k+"\"");p=ms.find(':',p)+1;return (int)strtol(ms.c_str()+p,0,10);};
  auto jf=[&](const char* k){auto p=ms.find(std::string("\"")+k+"\"");p=ms.find(':',p)+1;return strtof(ms.c_str()+p,0);};
  int STEPS=ji("steps"),HN=ji("harmonic_num"),US=ji("upsample_scale"),H=HN+1;
  float SR=jf("sampling_rate"),THR=jf("voiced_threshold"),SAMP=jf("sine_amp"),NSTD=jf("noise_std");
  std::vector<long> toks; {auto p=ms.find("\"tokens\"");p=ms.find('[',p)+1;auto e=ms.find(']',p);const char* c=ms.c_str()+p;
    while(c<ms.c_str()+e){char* nx;long v=strtol(c,&nx,10);if(nx==c){c++;continue;}toks.push_back(v);c=nx;while(*c==','||*c==' ')c++;}}
  int L=toks.size();
  CK(cudaStreamCreate(&stream));
  IRuntime* rt=createInferRuntime(gLogger);
  Eng A,B,C; A.load(rt,DIR+"A.plan"); B.load(rt,DIR+"B.plan"); C.load(rt,DIR+"C.plan");

  const int MB=16,MT=256,MF=512;  // max batch/tokens/frames
  // allocate max buffers (fp16 unless noted)
  auto H16=[&](const std::string& n,size_t el){ alloc(n,el*2);};
  alloc("tokens",(size_t)MB*MT*8); alloc("input_lengths",(size_t)MB*8);           // int64
  H16("ref_s",MB*256);H16("diff_noise",MB*256);H16("step_noise",(size_t)MB*(STEPS-1)*256);
  H16("num_steps",1);H16("alpha",1);H16("beta",1);
  H16("pred_dur",(size_t)MB*MT);H16("t_en",(size_t)MB*512*MT);H16("d",(size_t)MB*MT*640);H16("sty",MB*128);H16("ref",MB*128);
  H16("aln",(size_t)MB*MT*MF);
  H16("F0",(size_t)MB*2*MF);H16("N",(size_t)MB*2*MF);H16("asr",(size_t)MB*512*MF);
  alloc("phase",(size_t)MB*2*MF*H*8);   // fp64 (phase in cycles)
  H16("har",(size_t)MB*2*MF*US);H16("audio",(size_t)MB*2*MF*US);
  alloc("lin",(H+1)*4);                                                            // fp32 nsf weights

  // load nsf lin + fixed inputs -> gpu
  auto lin=readf32(DIR+"nsf_lin.bin"); CK(cudaMemcpy(buf["lin"],lin.data(),lin.size()*4,cudaMemcpyHostToDevice));
  auto ldF=[&](const std::string& f,const std::string& dst,int repeatB,int perB){ // fp32 file -> fp16 gpu, replicated over batch
    auto v=readf32(DIR+f); float* tmp; CK(cudaMalloc(&tmp,(size_t)repeatB*perB*4));
    for(int b=0;b<repeatB;b++) CK(cudaMemcpy(tmp+(size_t)b*perB,v.data(),perB*4,cudaMemcpyHostToDevice));
    kCastF2H<<<(repeatB*perB+255)/256,256,0,stream>>>(tmp,(half*)buf[dst],(size_t)repeatB*perB); CK(cudaStreamSynchronize(stream)); CK(cudaFree(tmp));};
  // rand_ini fp32 kept separately (host)
  auto rand_ini=readf32(DIR+"inp_rand_ini.bin"); float* d_ri; CK(cudaMalloc(&d_ri,(size_t)MB*H*4));

  // scalars
  { half hs[3]={__float2half((float)STEPS),__float2half(0.7f),__float2half(0.3f)};
    CK(cudaMemcpy(buf["num_steps"],&hs[0],2,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(buf["alpha"],&hs[1],2,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(buf["beta"],&hs[2],2,cudaMemcpyHostToDevice)); }
  // tokens int64 (replicated per batch below)
  std::vector<long> tkrow(toks);

  auto bind=[&](Eng& e){ for(auto&n:e.ins) e.ctx->setTensorAddress(n.c_str(),buf[n]); for(auto&n:e.outs) e.ctx->setTensorAddress(n.c_str(),buf[n]); };

  // -------- one pipeline pass for batch bs; returns F0 length (2F). If Ffix>0 skip host readback --------

  auto setShapesA=[&](int bs,int Lc){
    A.ctx->setInputShape("tokens",D({bs,Lc})); A.ctx->setInputShape("input_lengths",D({bs}));
    A.ctx->setInputShape("ref_s",D({bs,256})); A.ctx->setInputShape("diff_noise",D({bs,1,256}));
    A.ctx->setInputShape("step_noise",D({bs,STEPS-1,256}));
    Dims sc; sc.nbDims=0; A.ctx->setInputShape("num_steps",sc); A.ctx->setInputShape("alpha",sc); A.ctx->setInputShape("beta",sc);
  };

  // prepare per-batch inputs (tokens int64 replicated, il, noise replicated)
  auto prepInputs=[&](int bs){
    std::vector<long> tk((size_t)bs*L); for(int b=0;b<bs;b++) memcpy(&tk[(size_t)b*L],tkrow.data(),L*8);
    CK(cudaMemcpy(buf["tokens"],tk.data(),(size_t)bs*L*8,cudaMemcpyHostToDevice));
    std::vector<long> il(bs,L); CK(cudaMemcpy(buf["input_lengths"],il.data(),bs*8,cudaMemcpyHostToDevice));
    ldF("inp_ref_s.bin","ref_s",bs,256); ldF("inp_diff_noise.bin","diff_noise",bs,256); ldF("inp_step_noise.bin","step_noise",bs,(STEPS-1)*256);
    std::vector<float> rib((size_t)bs*H); for(int b=0;b<bs;b++) memcpy(&rib[(size_t)b*H],rand_ini.data(),H*4);
    CK(cudaMemcpy(d_ri,rib.data(),(size_t)bs*H*4,cudaMemcpyHostToDevice));
  };

  // set all input shapes + bind addresses for a given (bs, frames F)
  auto setShapes=[&](int bs,int F){
    setShapesA(bs,L);
    B.ctx->setInputShape("t_en",D({bs,512,L})); B.ctx->setInputShape("d",D({bs,L,640}));
    B.ctx->setInputShape("sty",D({bs,128})); B.ctx->setInputShape("aln",D({bs,L,F}));
    int twoF=2*F;
    C.ctx->setInputShape("asr",D({bs,512,F})); C.ctx->setInputShape("F0",D({bs,twoF}));
    C.ctx->setInputShape("N",D({bs,twoF})); C.ctx->setInputShape("ref",D({bs,128}));
    C.ctx->setInputShape("har",D({bs,1,twoF*US}));
    bind(A); bind(B); bind(C);
  };
  // GPU-only enqueue (capturable into a CUDA graph): A -> align -> B -> NSF har -> C
  auto gpuEnqueue=[&](int bs,int F){
    A.ctx->enqueueV3(stream);
    kFillH<<<((size_t)bs*L*F+255)/256,256,0,stream>>>((half*)buf["aln"],(size_t)bs*L*F,0.f);
    { dim3 g((L+127)/128,bs); kAlign<<<g,128,0,stream>>>((half*)buf["pred_dur"],(half*)buf["aln"],bs,L,F);}
    B.ctx->enqueueV3(stream);
    int twoF=2*F;
    { size_t tot=(size_t)bs*twoF*H; kRad<<<(tot+255)/256,256,0,stream>>>((half*)buf["F0"],d_ri,(double*)buf["phase"],bs,twoF,H,SR);
      dim3 g((H+31)/32,bs); kScan<<<g,32,0,stream>>>((double*)buf["phase"],bs,twoF,H,US);}
    { size_t T=(size_t)twoF*US; dim3 g((T+255)/256,bs); kHar<<<g,256,0,stream>>>((half*)buf["F0"],(double*)buf["phase"],(float*)buf["lin"],(half*)buf["har"],bs,twoF,H,US,THR,SAMP,NSTD);}
    C.ctx->enqueueV3(stream);
  };

  bool doDump = (argc>1 && std::string(argv[1])=="dump");
  int batches[]={1,2,4,8,16};
  for(int bi=0;bi<5;bi++){ int bs=batches[bi];
    prepInputs(bs);
    // pass 1: run A to determine data-dependent frame count F
    setShapesA(bs,L); bind(A); A.ctx->enqueueV3(stream); CK(cudaStreamSynchronize(stream));
    std::vector<half> pd((size_t)bs*L); CK(cudaMemcpy(pd.data(),buf["pred_dur"],(size_t)bs*L*2,cudaMemcpyDeviceToHost));
    int F=0; for(int b=0;b<bs;b++){int s=0; for(int l=0;l<L;l++) s+=(int)(__half2float(pd[(size_t)b*L+l])+0.5f); F=std::max(F,s);}
    int twoF=2*F; size_t T=(size_t)twoF*US;
    // set shapes for this F, warmup full pipeline (also initializes TRT contexts for capture)
    setShapes(bs,F); gpuEnqueue(bs,F); CK(cudaStreamSynchronize(stream));
    // finite check
    std::vector<half> au((size_t)bs*T); CK(cudaMemcpy(au.data(),buf["audio"],(size_t)bs*T*2,cudaMemcpyDeviceToHost));
    bool fin=true; float mn=1e9,mx=-1e9; for(size_t i=0;i<(size_t)bs*T;i++){float v=__half2float(au[i]); if(!isfinite(v))fin=false; mn=std::min(mn,v);mx=std::max(mx,v);}
    if(bi==0 && doDump){ auto dF=[&](const char* nm,const char* fn,size_t n){ std::vector<half> h(n); CK(cudaMemcpy(h.data(),buf[nm],n*2,cudaMemcpyDeviceToHost));
        std::vector<float> f(n); for(size_t i=0;i<n;i++)f[i]=__half2float(h[i]); FILE* fp=fopen(fn,"wb"); fwrite(f.data(),4,n,fp); fclose(fp);};
      dF("pred_dur","/tmp/cpp_pred_dur.bin",(size_t)bs*L); dF("F0","/tmp/cpp_F0.bin",(size_t)bs*2*F);
      dF("har","/tmp/cpp_har.bin",(size_t)bs*2*F*US); dF("audio","/tmp/cpp_audio.bin",(size_t)bs*2*F*US);
      { size_t n=(size_t)bs*2*F*H; std::vector<double> ph(n); CK(cudaMemcpy(ph.data(),buf["phase"],n*8,cudaMemcpyDeviceToHost)); FILE* fp=fopen("/tmp/cpp_phase.bin","wb"); fwrite(ph.data(),8,n,fp); fclose(fp);}
      { FILE* fp=fopen("/tmp/cpp_F.txt","w"); fprintf(fp,"%d\n",F); fclose(fp);} printf("  dumped validation tensors (F=%d)\n",F);
    }
    // capture full pipeline into a CUDA graph and replay for timing
    cudaGraph_t graph; cudaGraphExec_t gexec;
    CK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal)); gpuEnqueue(bs,F); CK(cudaStreamEndCapture(stream,&graph));
    CK(cudaGraphInstantiate(&gexec,graph,0));
    for(int w=0;w<20;w++) CK(cudaGraphLaunch(gexec,stream)); CK(cudaStreamSynchronize(stream));
    int N=200;
    // (1) CUDA-event timing
    cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    CK(cudaEventRecord(e0,stream)); for(int i=0;i<N;i++) CK(cudaGraphLaunch(gexec,stream)); CK(cudaEventRecord(e1,stream)); CK(cudaEventSynchronize(e1));
    float ev; CK(cudaEventElapsedTime(&ev,e0,e1));
    // (2) CPU wall-clock (sync-bounded) over the same N launches
    CK(cudaStreamSynchronize(stream));
    auto w0=std::chrono::high_resolution_clock::now();
    for(int i=0;i<N;i++) CK(cudaGraphLaunch(gexec,stream)); CK(cudaStreamSynchronize(stream));
    auto w1=std::chrono::high_resolution_clock::now();
    double wall=std::chrono::duration<double,std::milli>(w1-w0).count()/N;
    // (3) single-shot latency: sync, one launch, sync (no possible cross-iter overlap)
    double singleSum=0; int NS=30;
    for(int i=0;i<NS;i++){ CK(cudaStreamSynchronize(stream)); auto s0=std::chrono::high_resolution_clock::now();
      CK(cudaGraphLaunch(gexec,stream)); CK(cudaStreamSynchronize(stream)); auto s1=std::chrono::high_resolution_clock::now();
      singleSum+=std::chrono::duration<double,std::milli>(s1-s0).count(); }
    double single=singleSum/NS;
    CK(cudaGraphExecDestroy(gexec)); CK(cudaGraphDestroy(graph));
    printf("batch=%2d F=%d audio=%.2fs finite=%d rng[%.3f,%.3f]  event=%.3f  wall=%.3f  single-shot=%.3f ms/iter\n",
           bs,F,(double)T/24000.0,fin,mn,mx,ev/N,wall,single);
  }
  return 0;
}
