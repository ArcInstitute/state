import torch
import torch.nn as nn
import torch.nn.functional as F
from geomloss import SamplesLoss


class WassersteinLoss(nn.Module):
    """
    Implements Wasserstein distance loss for distributions represented by logits.
    This implementation supports both 1D and 2D Wasserstein distance calculations.
    """

    def __init__(self, p=1, reduction="mean"):
        """
        Args:
            p (int): Order of Wasserstein distance (1 or 2)
            reduction (str): 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.p = p
        self.reduction = reduction

    def forward(self, p, q):
        """
        Compute Wasserstein distance between predicted and target distributions.

        Args:
            logits (torch.Tensor): Predicted logits of shape (batch_size, num_classes)
            target (torch.Tensor): Target probabilities of shape (batch_size, num_classes)
                                 or class indices of shape (batch_size,)

        Returns:
            torch.Tensor: Computed Wasserstein distance
        """

        q = torch.nan_to_num(q, nan=0.0)
        # Convert logits to probabilities
        pred_probs = F.softmax(p, dim=-1)
        q = F.softmax(q, dim=-1)

        # Compute cumulative distribution functions (CDFs)
        pred_cdf = torch.cumsum(pred_probs, dim=-1)
        target_cdf = torch.cumsum(q, dim=-1)

        max_len = max(pred_cdf.size(1), target_cdf.size(1))
        if pred_cdf.size(1) < max_len:
            pred_cdf = F.pad(pred_cdf, (0, max_len - pred_cdf.size(1)), "constant", 0)
        if target_cdf.size(1) < max_len:
            target_cdf = F.pad(target_cdf, (0, max_len - target_cdf.size(1)), "constant", 0)

        # Compute Wasserstein distance
        wasserstein_dist = torch.abs(pred_cdf - target_cdf).pow(self.p)
        wasserstein_dist = wasserstein_dist.sum(dim=-1)

        # Apply reduction if specified
        if self.reduction == "mean":
            return wasserstein_dist.mean()
        elif self.reduction == "sum":
            return wasserstein_dist.sum()
        return wasserstein_dist


class KLDivergenceLoss(nn.Module):
    def __init__(self, apply_normalization=False, epsilon=1e-10):
        super().__init__()
        self.apply_normalization = apply_normalization
        self.epsilon = epsilon

    def forward(self, p, q):
        q = torch.nan_to_num(q, nan=0.0)
        p = torch.nan_to_num(p, nan=0.0)

        max_len = max(p.size(1), q.size(1))
        if p.size(1) < max_len:
            p = F.pad(p, (0, max_len - p.size(1)), "constant", 0)
        if q.size(1) < max_len:
            q = F.pad(q, (0, max_len - q.size(1)), "constant", 0)

        if self.apply_normalization:
            p = F.softmax(p, dim=-1)
            q = F.softmax(q, dim=-1)

        return torch.sum(p * torch.log(p / q))


class MMDLoss(nn.Module):
    def __init__(self, kernel="energy", blur=0.05, scaling=0.5, downsample=1):
        super().__init__()
        self.mmd_loss = SamplesLoss(loss=kernel, blur=blur, scaling=scaling)
        self.downsample = downsample

    def forward(self, input, target):
        input = input.reshape(-1, self.downsample, input.shape[-1])
        target = target.reshape(-1, self.downsample, target.shape[-1])

        loss = self.mmd_loss(input, target)
        return loss.mean()


class TabularLoss(nn.Module):
    def __init__(self, shared=128, downsample=1):
        super().__init__()
        self.shared = shared
        self.downsample = downsample

        self.gene_loss = SamplesLoss(loss="energy")
        self.cell_loss = SamplesLoss(loss="energy")

    def forward(self, input, target):
        input = input.reshape(-1, self.downsample, input.shape[-1])
        target = target.reshape(-1, self.downsample, target.shape[-1])
        gene_mmd = self.gene_loss(input, target).nanmean()

        # cell_mmd should only be on the shared genes, and match scale to mse loss
        cell_inputs = input[:, :, -self.shared :]
        cell_targets = target[:, :, -self.shared :]

        # need to reshape each from (B, self.downsample, F) to (F, self.downsample, B)
        cell_inputs = cell_inputs.transpose(2, 0)
        cell_targets = cell_targets.transpose(2, 0)
        cell_mmd = self.cell_loss(cell_inputs, cell_targets).nanmean()

        final_loss = torch.tensor(0.0).to(cell_mmd.device)
        if not gene_mmd.isnan():
            final_loss += gene_mmd
        if not cell_mmd.isnan():
            final_loss += cell_mmd

        return final_loss
