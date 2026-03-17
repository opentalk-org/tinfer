import os
import yaml
import torch
from transformers import AlbertConfig, AlbertModel

class CustomAlbert(AlbertModel):
    def forward(self, *args, **kwargs):
        # Call the original forward method
        outputs = super().forward(*args, **kwargs)

        # Only return the last_hidden_state
        return outputs.last_hidden_state

def build_plbert(config: dict):
    albert_base_configuration = AlbertConfig(**config)
    bert = CustomAlbert(albert_base_configuration)
    return bert

def load_plbert(bert, plbert_path: str):
    ckpts = []# Only return the last_hidden_state
    for f in os.listdir(plbert_path):
        if f.startswith("step_"): ckpts.append(f)

    iters = [int(f.split('_')[-1].split('.')[0]) for f in ckpts if os.path.isfile(os.path.join(plbert_path, f))]
    iters = sorted(iters)[-1]

    checkpoint = torch.load(plbert_path + "/step_" + str(iters) + ".t7", map_location='cpu')
    state_dict = checkpoint['net']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        if name.startswith('encoder.'):
            name = name[8:] # remove `encoder.`
            new_state_dict[name] = v
    del new_state_dict["embeddings.position_ids"]
    bert.load_state_dict(new_state_dict, strict=False)
    
    return bert
