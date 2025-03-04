import logging
import numpy as np
import torch
import torch.nn as nn

from sklearn.decomposition import PCA
from typing import Optional, Union

logger = logging.getLogger(__name__)


class PCATransform(nn.Module):
    """
    A PCA transform that:
      - Uses scikit-learn's PCA for fitting on CPU.
      - Stores learned buffers (mean_, components_, singular_values_) in float32 torch tensors.
      - Allows subsequent encoding/decoding on any device (CPU/GPU).
    """

    def __init__(self, n_components: int = 200, device: Optional[Union[str, torch.device]] = "cpu"):
        super().__init__()
        self.n_components = n_components

        # Use PyTorch buffers so they move automatically with .to(device).
        self.register_buffer("mean_", None)
        self.register_buffer("components_", None)
        self.register_buffer("singular_values_", None)

        # Keep track of current device explicitly
        self.device = torch.device("cpu")

    def name(self) -> str:
        return "PCATransform"

    def fit(self, X: Union[np.ndarray, torch.Tensor]) -> "PCATransform":
        """
        Fit the PCA transform on CPU using scikit-learn, then store
        the results in float32 PyTorch buffers.

        Args:
            X: Data matrix of shape (n_samples, n_features).

        Returns:
            self
        """
        # 1. Move data to CPU and convert to float32 numpy
        #    (Scikit-learn typically expects double precision, but we can
        #     force float32 to reduce memory usage; scikit-learn can handle it.)
        if isinstance(X, torch.Tensor):
            # Move to CPU and convert to numpy
            X_cpu = X.to("cpu").detach().float().numpy().astype(np.float32)
        else:
            # Already numpy; just ensure float32
            X_cpu = X.astype(np.float32, copy=False)

        logger.info(
            f"Fitting scikit-learn PCA with n_components={self.n_components} "
            f"on CPU (data shape={X_cpu.shape}, dtype={X_cpu.dtype})."
        )

        # 2. Fit scikit-learn PCA (single pass).
        pca = PCA(n_components=self.n_components, svd_solver="auto")
        pca.fit(X_cpu)

        # 3. Store results as float32 in PyTorch buffers
        self.mean_ = torch.tensor(pca.mean_, dtype=torch.float32)
        self.components_ = torch.tensor(pca.components_, dtype=torch.float32)  # shape: (n_components, n_features)
        self.singular_values_ = torch.tensor(pca.singular_values_, dtype=torch.float32)

        # 4. Print explained variance ratio
        explained_var_ratio_cum = np.sum(pca.explained_variance_ratio_)
        logger.info(
            f"PCA fit complete. Retained {self.n_components} components. "
            f"Cumulative explained variance ratio: {explained_var_ratio_cum:.3f}"
        )

        return self

    @torch.no_grad()
    def encode(self, X: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Project/encode data into PCA space.

        Args:
            X: Data of shape (n_samples, n_features).

        Returns:
            Projected data of shape (n_samples, n_components).
        """
        if self.components_ is None:
            raise RuntimeError("PCATransform not fitted. Call fit() first.")

        # Convert input to the same device as self.components_
        if isinstance(X, np.ndarray):
            X_torch = torch.from_numpy(X.astype(np.float32, copy=False))
        else:
            X_torch = X

        # Make sure it's on the correct device
        X_torch = X_torch.to(self.device)

        # Center the data
        X_centered = X_torch - self.mean_

        # (N, D) x (D, K)^T => (N, K)
        # But self.components_ is (K, D), so we do (N, D) x (K, D).T => (N, K)
        return torch.matmul(X_centered, self.components_.T)

    @torch.no_grad()
    def decode(self, X: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Decode from PCA space back to the original feature space.

        Args:
            X: Data in PCA space of shape (n_samples, n_components).

        Returns:
            Reconstructed data of shape (n_samples, n_features).
        """
        if self.components_ is None:
            raise RuntimeError("PCATransform not fitted. Call fit() first.")

        # Convert input to the same device as self.components_
        if isinstance(X, np.ndarray):
            X_torch = torch.from_numpy(X.astype(np.float32, copy=False))
        else:
            X_torch = X

        X_torch = X_torch.to(self.device)

        # (N, K) x (K, D) => (N, D)
        reconstructed = torch.matmul(X_torch, self.components_)
        return reconstructed + self.mean_

    def to(self, device):
        """
        Override .to(device) to move buffers and update self.device.
        """
        super().to(device)
        self.device = torch.device(device)
        return self
