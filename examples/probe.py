import sys; sys.path.insert(0,"/workspace/tinfer/tinfer")
import torch
from tinfer.models.impl.styletts2.model.modules.load_utils import load_model
m,_=load_model("/workspace/converted_models/libri/model.pth")
for k in m: m[k]=m[k].cuda().eval()
net=m.diffusion.diffusion.net
for B in (1,2):
    L=171
    x=torch.randn(B,1,256).cuda(); c_noise=torch.randn(B).cuda()
    emb=torch.randn(B,L,768).cuda(); feat=torch.randn(B,256).cuda()
    with torch.no_grad():
        out=net(x, c_noise, embedding=emb, features=feat)
    print(f"B={B}: net out {tuple(out.shape)}")
