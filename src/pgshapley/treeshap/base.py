from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .unified import UnifiedEnsemble


@dataclass
class PreparedModel:
    """Backend-specific precomputation result."""

    ensemble: UnifiedEnsemble
    expected_value: np.ndarray  # (n_outputs,)


class TreeShapBackend(ABC):
    """Backend interface for unified tree models."""

    def __init__(self):
        self.prepared: Optional[PreparedModel] = None

    @abstractmethod
    def prepare(self, ensemble: UnifiedEnsemble) -> PreparedModel:
        """Precompute everything that does not depend on the sample(s)."""

    @abstractmethod
    def explain(self, X: np.ndarray, *, tree_limit: Optional[int] = None) -> np.ndarray:
        """Explain samples.

        Parameters
        ----------
        X:
            Array of shape (n_samples, n_features).

        Returns
        -------
        shap_values:
            Array of shape (n_samples, n_features, n_outputs).
        """
