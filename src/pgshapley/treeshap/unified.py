from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass(frozen=True)
class UnifiedTree:
    """A unified representation of a single binary decision tree.

    This representation is intentionally close to scikit-learn's ``tree_`` arrays
    while being model-library agnostic.

    Notes
    -----
    * Internal nodes have ``feature[node] >= 0``.
    * Leaf nodes have ``feature[node] == -2`` (sklearn convention) and
      ``children_left[node] == children_right[node] == -1``.
    * ``values[node]`` is the model output *at that node* as a vector of length
      ``n_outputs``. For classifiers this is typically probabilities.
    * ``node_weight[node]`` is the (weighted) number of training samples reaching
      that node. Edge weights for missing-value routing are derived as
      ``node_weight[child]/node_weight[parent]``.
    """

    children_left: np.ndarray  # (n_nodes,)
    children_right: np.ndarray  # (n_nodes,)
    feature: np.ndarray  # (n_nodes,)
    threshold: np.ndarray  # (n_nodes,)
    values: np.ndarray  # (n_nodes, n_outputs)
    node_weight: np.ndarray  # (n_nodes,)

    # Metadata
    n_features: int
    n_outputs: int


@dataclass(frozen=True)
class UnifiedEnsemble:
    """A unified representation of a tree or a tree ensemble."""

    trees: List[UnifiedTree]
    tree_weights: np.ndarray  # (n_trees,)

    n_features: int
    n_outputs: int

    model_type: str  # e.g. "sklearn"
    objective: str  # e.g. "regression" | "classification"

    # Optional names (kept for SHAP drop-in compatibility)
    feature_names: Optional[List[str]] = None
