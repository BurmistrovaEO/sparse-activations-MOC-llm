"""Backward-compatible shim for misspelled module name.

Prefer importing from `sparsification.py`.
"""

from sparsification import SparseMLP, replace_nested_module

__all__ = ["SparseMLP", "replace_nested_module"]
