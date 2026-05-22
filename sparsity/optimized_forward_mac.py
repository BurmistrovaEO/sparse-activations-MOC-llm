import torch
import torch.nn as nn
from typing import Dict, List


class SparseGateFunction(torch.autograd.Function):
    """Custom autograd function that reuses sparse activations from forward"""

    @staticmethod
    def forward(ctx, x, gate_proj, up_proj, down_proj, act_fn, k):
        # Forward sparse computation
        gate_out = gate_proj(x)
        up_out = up_proj(x)

        # Create sparse mask
        _, top_indices = torch.topk(gate_out, k, dim=-1, largest=True)
        mask = torch.zeros_like(gate_out)
        mask.scatter_(dim=-1, index=top_indices, value=1.0)

        # Sparse activations
        gate_sparse = gate_out * mask
        up_sparse = up_out * mask
        act_sparse = act_fn(gate_sparse)
        product_sparse = act_sparse * up_sparse

        output = down_proj(product_sparse)

        # Save only sparse activations (no weights!)
        ctx.save_for_backward(
            x,
            gate_sparse,
            up_sparse,
            act_sparse,
            product_sparse,
            mask,
            top_indices
        )
        ctx.act_fn = act_fn

        # Save module references for backward
        ctx.gate_proj = gate_proj
        ctx.up_proj = up_proj
        ctx.down_proj = down_proj

        return output

    @staticmethod
    def backward(ctx, grad_output):
        _, gate_sparse, up_sparse, act_sparse, __, mask, ___ = (
            ctx.saved_tensors
        )

        # Get modules and their weights
        gate_proj = ctx.gate_proj
        up_proj = ctx.up_proj
        down_proj = ctx.down_proj

        gate_weight = gate_proj.weight
        up_weight = up_proj.weight
        down_weight = down_proj.weight

        grad_product = grad_output.mm(down_weight)
        grad_act = grad_product * up_sparse
        grad_up = grad_product * act_sparse

        # SILU derivative
        sig = torch.sigmoid(gate_sparse)
        grad_gate = grad_act * (sig * (1 + gate_sparse * (1 - sig)))

        # Backprop through mask
        grad_gate_dense = grad_gate * mask
        grad_up_dense = grad_up * mask

        # Input gradient
        grad_x = grad_gate_dense.mm(gate_weight) + grad_up_dense.mm(up_weight)

        return (grad_x, None, None, None, None, None)


class SparseMLP(nn.Module):
    """
    Sparse MLP that can replace a standard MLP module
    Compatible with standard nn.Linear layers
    """

    def __init__(self, base_layer, k=4096, n=None, m=None):
        super().__init__()

        self.config = base_layer.config
        self.hidden_size = base_layer.config.hidden_size
        self.intermediate_size = base_layer.config.intermediate_size

        self.gate_proj = base_layer.gate_proj
        self.up_proj = base_layer.up_proj
        self.down_proj = base_layer.down_proj
        self.act_fn = base_layer.act_fn
        self.n = None
        self.m = None
        self.k = k

        # Sparse MLP layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return SparseGateFunction.apply(
            x,
            self.gate_proj,
            self.up_proj,
            self.down_proj,
            self.act_fn,
            self.k,
            self.n,
            self.m,
        )


# ============ MODEL REPLACEMENT UTILITIES ============


def find_mlp_modules(
    model: nn.Module, module_types: List[type] = None
) -> Dict[str, nn.Module]:
    """
    Find all MLP-like modules in a model

    Args:
        model: PyTorch model
        module_types: List of module types to look for (default: [nn.Linear])

    Returns:
        Dictionary mapping path to module
    """
    if module_types is None:
        module_types = [nn.Linear]

    mlp_modules = {}

    def scan_module(module, path=""):
        # Check if this is a linear layer that could be part of MLP
        for module_type in module_types:
            if isinstance(module, module_type):
                mlp_modules[path] = module
                break

        # Recursively scan children
        for name, child in module.named_children():
            new_path = f"{path}.{name}" if path else name
            scan_module(child, new_path)

    scan_module(model)
    return mlp_modules
