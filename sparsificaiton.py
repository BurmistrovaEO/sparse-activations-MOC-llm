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
        self.k_min = 1
        if not (self.k_min <= k <= self.k_max):
            raise ValueError(f"k must be in [{self.k_min}, {self.k_max}], got {k}")
        self.k = k

    def forward(self, x):
        gate_out = self.gate_proj(x)
        _, indices = torch.topk(gate_out, self.k, dim=-1, largest=True)
        mask = torch.zeros_like(gate_out, dtype=torch.bool)
        mask.scatter_(dim=-1, index=indices, value=True)

        gate_out = gate_out * mask

        up_out = self.up_proj(x)
        up_out = up_out * mask

        act_out = self.act_fn(gate_out)

        down_proj = self.down_proj(act_out * up_out)
        return down_proj

def replace_nested_module(model, name, new_module):
    parts = name.split('.')
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)
