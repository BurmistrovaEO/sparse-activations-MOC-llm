import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import copy

# ============ SPARSE MLP MODULES THAT ACCEPT ORIGINAL MODULE ============


class TritonSparseMLP(nn.Module):
    """
    Sparse MLP that can be instantiated from an existing MLP module
    Copies architecture, dimensions, and weights from the original
    """

    def __init__(self, original_module: nn.Module, sparsity_ratio: float = 0.1,
                 k: Optional[int] = None, use_triton: bool = True,
                 n: int = None, m: int = None):
        """
        Create a sparse MLP from an existing MLP module

        Args:
            original_module: The original MLP module to replace
                             (e.g., FFN in transformer)
            sparsity_ratio: Fraction of neurons to keep (e.g., 0.1 = keep 10%)
            k: Number of top neurons to keep
                            (overrides sparsity_ratio if provided)
            use_triton: Whether to use Triton kernels (if available)
        """
        super().__init__()

        self.use_triton = use_triton and torch.cuda.is_available()

        # Detect MLP structure from original module
        self._detect_mlp_structure(original_module)

        # Calculate k (number of neurons to keep)
        if k is None:
            k = int(self.hidden_dim * sparsity_ratio)
        self.k = max(1, min(k, self.hidden_dim))
        self.n = None
        self.m = None
        self.sparsity_ratio = sparsity_ratio

        # Create sparse projections
        self.gate_proj = nn.Linear(self.in_features,
                                   self.hidden_dim,
                                   bias=self.has_gate_bias)
        self.up_proj = nn.Linear(self.in_features,
                                 self.hidden_dim,
                                 bias=self.has_up_bias)
        self.down_proj = nn.Linear(self.hidden_dim,
                                   self.out_features,
                                   bias=self.has_down_bias)
        self.act_fn = self.activation_fn

        # Copy weights from original module
        self._copy_weights(original_module)

        # Register buffers for sparse indices and masks (for debugging)
        self.register_buffer('last_indices', None)
        self.register_buffer('last_mask', None)

    def _detect_mlp_structure(self, module: nn.Module):
        """Detect the structure of the original MLP module"""

        # Try to detect common MLP patterns
        self.in_features = None
        self.hidden_dim = None
        self.out_features = None
        self.has_gate_bias = True
        self.has_up_bias = True
        self.has_down_bias = True
        self.activation_fn = nn.SiLU()  # Default

        # Pattern 1: Standard LLM FFN with gate_proj, up_proj, down_proj
        if (
            hasattr(module, 'gate_proj')
            and hasattr(module, 'up_proj')
            and hasattr(module, 'down_proj')
        ):
            self.gate_proj_orig = module.gate_proj
            self.up_proj_orig = module.up_proj
            self.down_proj_orig = module.down_proj
            self.in_features = module.gate_proj.in_features
            self.hidden_dim = module.gate_proj.out_features
            self.out_features = module.down_proj.out_features
            self.has_gate_bias = module.gate_proj.bias is not None
            self.has_up_bias = module.up_proj.bias is not None
            self.has_down_bias = module.down_proj.bias is not None

        # Pattern 2: Sequential MLP (common in some architectures)
        elif isinstance(module, nn.Sequential):
            linear_layers = [m for m in module if isinstance(m, nn.Linear)]
            if len(linear_layers) >= 2:
                self.in_features = linear_layers[0].in_features
                self.hidden_dim = linear_layers[0].out_features
                self.out_features = linear_layers[-1].out_features

                # Store original layers for weight copying
                self.gate_proj_orig = linear_layers[0]
                self.up_proj_orig = (
                                        linear_layers[1]
                                        if len(linear_layers) > 1
                                        else None
                                    )
                self.down_proj_orig = linear_layers[-1]

                # Find activation function
                for m in module:
                    if isinstance(m, (nn.GELU, nn.ReLU, nn.SiLU)):
                        self.activation_fn = m
                        break

        # Pattern 3: Custom MLP with fc1, fc2 pattern
        elif hasattr(module, 'fc1') and hasattr(module, 'fc2'):
            self.gate_proj_orig = module.fc1
            self.down_proj_orig = module.fc2
            self.in_features = module.fc1.in_features
            self.hidden_dim = module.fc1.out_features
            self.out_features = module.fc2.out_features

            # Check for up_proj (some have gate and up)
            if hasattr(module, 'fc3'):
                self.up_proj_orig = module.fc3

        # Pattern 4: LLaMA/Mistral style MLP
        elif (
            hasattr(module, 'w1')
            and hasattr(module, 'w2')
            and hasattr(module, 'w3')
        ):
            self.gate_proj_orig = module.w1
            self.up_proj_orig = module.w3
            self.down_proj_orig = module.w2
            self.in_features = module.w1.in_features
            self.hidden_dim = module.w1.out_features
            self.out_features = module.w2.out_features

        else:
            raise ValueError(f"Cannot detect MLP structure "
                             f"in module of type {type(module)}")

        # Ensure we have all required projections
        if not hasattr(self, 'gate_proj_orig'):
            raise ValueError("Could not identify gate "
                             "projection in original module")
        if not hasattr(self, 'down_proj_orig'):
            raise ValueError("Could not identify down "
                             "projection in original module")

    def _copy_weights(self, original_module: nn.Module):
        """Copy weights from original module to sparse module"""
        with torch.no_grad():
            # Copy gate projection weights
            if hasattr(self, 'gate_proj_orig'):
                self.gate_proj.weight.copy_(self.gate_proj_orig.weight)
                if (
                    self.gate_proj.bias is not None
                    and self.gate_proj_orig.bias is not None
                ):
                    self.gate_proj.bias.copy_(self.gate_proj_orig.bias)

            # Copy up projection weights
            if hasattr(self, 'up_proj_orig') and self.up_proj_orig is not None:
                self.up_proj.weight.copy_(self.up_proj_orig.weight)
                if (
                    self.up_proj.bias is not None
                    and self.up_proj_orig.bias is not None
                ):
                    self.up_proj.bias.copy_(self.up_proj_orig.bias)
            else:
                # If no separate up_proj, use gate_proj weights
                self.up_proj.weight.copy_(self.gate_proj.weight)
                if self.up_proj.bias is not None:
                    self.up_proj.bias.copy_(self.gate_proj.bias)

            # Copy down projection weights
            self.down_proj.weight.copy_(self.down_proj_orig.weight)
            if (
                self.down_proj.bias is not None
                and self.down_proj_orig.bias is not None
            ):
                self.down_proj.bias.copy_(self.down_proj_orig.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with sparsity"""
        if self.use_triton:
            return self._forward_triton(x)
        else:
            return self._forward_pytorch(x)

    def _forward_pytorch(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch native forward pass (fallback)"""
        # Compute gate and up projections
        gate_out = self.gate_proj(x)
        up_out = self.up_proj(x)

        # Top-k sparsity
        _, top_indices = torch.topk(gate_out, self.k, dim=-1)
        mask = torch.zeros_like(gate_out)
        mask.scatter_(-1, top_indices, 1.0)

        # Store for debugging
        self.last_indices = top_indices
        self.last_mask = mask

        # Apply sparsity
        gate_sparse = gate_out * mask
        up_sparse = up_out * mask

        # Activation and product
        act_sparse = self.act_fn(gate_sparse)
        product_sparse = act_sparse * up_sparse

        # Down projection
        return self.down_proj(product_sparse)

    def _forward_triton(self, x: torch.Tensor) -> torch.Tensor:
        """Triton-optimized forward pass"""
        # For now, fall back to PyTorch (Triton kernels need implementation)
        # In production, you would implement the Triton kernel here
        return self._forward_pytorch(x)


class SparseMLPWrapper(nn.Module):
    """
    Wrapper that makes any MLP sparse by applying
    top-k sparsity to its intermediate activations
    """

    def __init__(self, original_module: nn.Module,
                 sparsity_ratio: float = 0.1,
                 k: Optional[int] = None,
                 sparse_gradients: bool = False):
        """
        Wrap an existing MLP module to make it sparse

        Args:
            original_module: The original MLP module
            sparsity_ratio: Fraction of neurons to keep
            k: Number of top neurons to keep
            sparse_gradients: Whether to also sparsify gradients
        """
        super().__init__()
        self.original_module = original_module
        self.sparsity_ratio = sparsity_ratio
        self.sparse_gradients = sparse_gradients

        # Detect hidden dimension
        self.hidden_dim = self._detect_hidden_dim()

        if k is None:
            k = int(self.hidden_dim * sparsity_ratio)
        self.k = max(1, k)

    def _detect_hidden_dim(self) -> int:
        """Detect the hidden dimension of the MLP"""
        # Try common patterns
        if hasattr(self.original_module, 'gate_proj'):
            return self.original_module.gate_proj.out_features
        elif hasattr(self.original_module, 'fc1'):
            return self.original_module.fc1.out_features
        elif hasattr(self.original_module, 'w1'):
            return self.original_module.w1.out_features
        elif isinstance(self.original_module, nn.Sequential):
            for m in self.original_module:
                if isinstance(m, nn.Linear):
                    return m.out_features
        return 4096  # Default for LLMs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with sparsity applied to intermediate activations"""

        # Need to hook into the forward pass to apply sparsity
        # This is more complex - requires modifying the forward pass
        # For now, we assume the module has a specific structure

        if (
            hasattr(self.original_module, 'gate_proj')
            and hasattr(self.original_module, 'up_proj')
        ):
            # Standard pattern: compute gate and up separately
            gate_out = self.original_module.gate_proj(x)
            up_out = self.original_module.up_proj(x)

            # Apply sparsity
            _, top_indices = torch.topk(gate_out, self.k, dim=-1)
            mask = torch.zeros_like(gate_out)
            mask.scatter_(-1, top_indices, 1.0)

            gate_sparse = gate_out * mask
            up_sparse = up_out * mask

            # Continue with activation
            if hasattr(self.original_module, 'act_fn'):
                act_out = self.original_module.act_fn(gate_sparse)
            else:
                act_out = F.silu(gate_sparse)  # Default

            product = act_out * up_sparse
            return self.original_module.down_proj(product)

        else:
            # Fallback: just call original (no sparsity applied)
            return self.original_module(x)


# ============ MODEL REPLACEMENT UTILITIES ============

def replace_mlp_modules(
    model: nn.Module,
    sparsity_ratio: float = 0.1,
    target_names: Optional[List[str]] = None,
    module_type: type = None,
    use_triton: bool = True
) -> nn.Module:
    """
    Replace all MLP modules in a model with sparse versions

    Args:
        model: The model to modify
        sparsity_ratio: Fraction of neurons to keep (0.1 = keep 10%)
        target_names: List of module names to replace (if None, auto-detect)
        module_type: Specific module type to replace (e.g., FFN, MLP)
        use_triton: Use Triton kernels if available

    Returns:
        Model with sparse MLP modules
    """

    def replace_module(module, name=""):
        """Recursively replace modules"""

        # Check if this module should be replaced
        should_replace = False

        if target_names is not None:
            should_replace = any(
                target_name in name
                for target_name in target_names
            )
        elif module_type is not None:
            should_replace = isinstance(module, module_type)
        else:
            # Auto-detect MLP modules
            should_replace = (
                (hasattr(module, 'gate_proj')
                 and hasattr(module, 'up_proj')
                 and hasattr(module, 'down_proj')) or
                (hasattr(module, 'w1')
                 and hasattr(module, 'w2')
                 and hasattr(module, 'w3')) or
                (hasattr(module, 'fc1')
                 and hasattr(module, 'fc2'))
            )

        if should_replace:
            print(f"Replacing module: {name} (type: {type(module).__name__})")
            return TritonSparseMLP(
                module,
                sparsity_ratio=sparsity_ratio,
                use_triton=use_triton
            )

        # Recursively process children
        for child_name, child in module.named_children():
            new_child = replace_module(child, f"{name}.{child_name}"
                                       if name
                                       else child_name)
            if new_child is not None:
                setattr(module, child_name, new_child)

        return None

    # Make a deep copy to avoid modifying original
    model = copy.deepcopy(model)
    replace_module(model)

    return model


def convert_llama_mlp(model: nn.Module,
                      sparsity_ratio: float = 0.1) -> nn.Module:
    """
    Specifically convert LLaMA/Mistral style MLPs to sparse version
    """

    def convert_mlp(module):
        if (
            hasattr(module, 'w1')
            and hasattr(module, 'w2')
            and hasattr(module, 'w3')
        ):
            print(f"Converting MLP: {type(module).__name__}")
            return TritonSparseMLP(module, sparsity_ratio=sparsity_ratio)
        return None

    model = copy.deepcopy(model)

    def recursive_convert(module):
        for name, child in module.named_children():
            new_child = convert_mlp(child)
            if new_child is not None:
                setattr(module, name, new_child)
            else:
                recursive_convert(child)

    recursive_convert(model)
    return model


# ============ EXAMPLE USAGE ============

class LlamaMLP(nn.Module):
    """LLaMA-style MLP"""
    def __init__(self, dim: int = 4096, hidden_dim: int = 11008):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    """Simple transformer block"""
    def __init__(self, dim: int = 4096):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads=32)
        self.mlp = LlamaMLP(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + self.attention(x, x, x)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class LargeLanguageModel(nn.Module):
    """Example LLM with multiple transformer blocks"""
    def __init__(self, num_layers: int = 32, dim: int = 4096):
        super().__init__()
        self.embed_tokens = nn.Embedding(32000, dim)
        self.layers = nn.ModuleList([TransformerBlock(dim)
                                     for _ in range(num_layers)])
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, 32000, bias=False)

    def forward(self, input_ids):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)


if __name__ == "__main__":
    print("="*70)
    print("Sparse MLP Module Replacement for LLMs")
    print("="*70)

    # Create a large model
    print("\n1. Creating original model...")
    model = LargeLanguageModel(num_layers=4, dim=512)  # Small for testing
    print(f"   Total parameters: "
          f"{sum(p.numel() for p in model.parameters()):,}")

    # Count MLP modules
    mlp_count = sum(1 for _ in model.modules()
                    if hasattr(_, 'w1') and hasattr(_, 'w2'))
    print(f"   Found {mlp_count} MLP modules")

    # Replace with sparse MLPs
    print("\n2. Replacing MLPs with sparse versions...")
    sparse_model = replace_mlp_modules(
        model,
        sparsity_ratio=0.1,  # Keep only 10% of neurons
        use_triton=False  # Use PyTorch fallback for testing
    )

    print(f"   New parameters: "
          f"{sum(p.numel() for p in sparse_model.parameters()):,}")

    # Test forward/backward
    print("\n3. Testing forward/backward...")
    input_ids = torch.randint(0, 32000, (2, 128))
    output = sparse_model(input_ids)
    loss = output.sum()
    loss.backward()

    print(f"   ✓ Forward pass successful, output shape: {output.shape}")
    print("   ✓ Backward pass successful")

    # Calculate theoretical speedup
    hidden_dim = 2048  # Llama 7B hidden size
    k = int(hidden_dim * 0.1)
    print("\n4. Theoretical speedup:")
    print(f"   Original hidden dim: {hidden_dim}")
    print(f"   Sparse keep: {k} neurons ({0.1:.0%})")
    print(f"   Expected FLOPs reduction: ~{1 - k/hidden_dim:.0%}")

    # Example: Replace specific layers only
    print("\n5. Replacing specific layers...")
    sparse_model_v2 = convert_llama_mlp(model, sparsity_ratio=0.2)
    print("   ✓ Converted LLaMA-style MLPs with 20% sparsity")
