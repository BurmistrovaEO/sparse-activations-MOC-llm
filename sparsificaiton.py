import torch
from transformers.models.llama.modeling_llama import LlamaMLP
from transformers.activations import ACT2FN
import torch.nn as nn

class SparseMLP(nn.Module):
    def __init__(self, base_layer, k=4096):
        super().__init__()
        self.config = base_layer.config
        self.hidden_size = base_layer.config.hidden_size
        self.intermediate_size = base_layer.config.intermediate_size
        self.gate_proj = base_layer.gate_proj
        self.up_proj = base_layer.up_proj
        self.down_proj = base_layer.down_proj
        self.act_fn = base_layer.act_fn
        self.k_max = self.up_proj.in_features
        self.k_min = self.up_proj.in_features//2
        #TODO: check boundary and throw exception if needed
        self.k = k
        self.G_und = None
        self.U_und = None
        self.S_und = None

    def forward(self, x):
        gate_out = self.gate_proj(x)
        _, indices = torch.topk(gate_out, self.k, dim=-1, largest=True)
        mask = torch.zeros_like(gate_out, dtype=torch.bool)
        mask.scatter_(dim=-1, index=indices, value=True)

        gate_out = gate_out * mask
        self.G_und = gate_out.clone()

        up_out = self.up_proj(x)
        up_out = up_out * mask
        self.U_und = up_out.clone()

        act_out = self.act_fn(gate_out)
        act_out = act_out * mask
        self.S_und = act_out.clone()

        down_proj = self.down_proj(act_out * up_out)
        return down_proj
    
class NMsparseMLP(nn.Module):
    def __init__(self, base_layer, m = 2, n = 8):
        super().__init__()
        self.config = base_layer.config
        self.hidden_size = base_layer.config.hidden_size
        self.intermediate_size = base_layer.config.intermediate_size
        self.gate_proj = base_layer.gate_proj
        self.up_proj = base_layer.up_proj
        self.down_proj = base_layer.down_proj
        self.act_fn = base_layer.act_fn
        self.m = m
        self.n = n

    def forward(self, x):
        gate_out = self.gate_proj(x)
        sh0, sh1, sh2 = gate_out.shape
        gate_out = gate_out.reshape(sh0 * sh1 * (sh2//self.n), self.n)
        _, indices = torch.topk(gate_out, self.m, dim=-1, largest=True)
        mask = torch.zeros_like(gate_out, dtype=torch.bool)
        mask.scatter_(dim=-1, index=indices, value=True)

        gate_out = gate_out * mask

        mask = mask.reshape(sh0, sh1, sh2)

        up_out = self.up_proj(x)
        up_out = up_out * mask

        gate_out = gate_out.reshape(sh0, sh1, sh2)

        act_out = self.act_fn(gate_out)
        act_out = act_out * mask

        down_proj = self.down_proj(act_out * up_out)
        return down_proj

def replace_nested_module(model, name, new_module):
    parts = name.split('.')
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)
