from sparsity.sparsification import (SparseMLP,
                                     NMsparseMLP,
                                     replace_nested_module)
from sparsity.optimized_forward_mac import SparseMLP as SParseFWMLP
from transformers.models.llama.modeling_llama import LlamaMLP
from sparsity.triton_sparsity import TritonSparseMLP
import math

IMPORTANCE_RANKING = [27, 26, 25, 24, 23, 22,
                      21, 20, 19, 18, 15, 17,
                      16, 14, 13, 12, 11, 10,
                      9, 7, 8, 6, 5, 4, 3,
                      1, 2, 0]
BEGIN = [0, 1, 2, 3, 4, 5]
END = [22, 23, 24, 25, 26, 27]
MIDDLE = [11, 12, 13, 14, 15, 16]
BEGEND = [0, 1, 2, 25, 26, 27]


def set_up_sparsification(
    model,
    ablation_kind,
    importance_perc,
    sparse_implementation,
    device,
    k,
    n,
    m
):
    if ablation_kind == "importance":
        assert 0 <= importance_perc <= 100
        selected_amount = (len(IMPORTANCE_RANKING) * importance_perc) / 100
        selected_amount = math.ceil(selected_amount)
        sparse_indexes = IMPORTANCE_RANKING[:selected_amount]
    elif ablation_kind == "begin":
        sparse_indexes = BEGIN
    elif ablation_kind == "end":
        sparse_indexes = END
    elif ablation_kind == "begend":
        sparse_indexes = BEGEND
    else:
        sparse_indexes = MIDDLE

    if sparse_implementation == "vanilla_k":
        sparse_class = SparseMLP
    elif sparse_implementation == "vanilla_nm":
        sparse_class = NMsparseMLP
    elif sparse_implementation == "mps":
        sparse_class = SParseFWMLP
    else:
        sparse_class = TritonSparseMLP

    to_replace_names_modules = {}

    counter = 0
    for name, module in model.named_modules():
        if isinstance(module, LlamaMLP):
            if counter in sparse_indexes:
                to_replace_names_modules[name] = module
            counter += 1

    for name, module in to_replace_names_modules.items():
        sparseBlock = sparse_class(module, k=k, n=n, m=m)
        replace_nested_module(model, name, sparseBlock)

    return model.to(device)
