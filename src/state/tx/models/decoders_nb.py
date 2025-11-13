# models/decoders_nb.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import NegativeBinomial


class NBDecoder(nn.Module):
    """
    scVI‑style decoder that maps a latent embedding (optionally with batch covariates)
    to the parameters of a negative‑binomial (or ZINB) distribution over raw counts.

    Y_ig ~ NB(μ_ig, θ_g)         where
      μ_ig = l_i * softplus(W_g z_i + b_g)
      θ_g  = softplus(r_g)       (gene‑specific inverse dispersion)

    Optionally, a zero‑inflation gate π_ig can be produced (not shown here).
    """

    def __init__(
        self,
        latent_dim: int,
        gene_dim: int,
        hidden_dims=[1024, 256, 256],
        dropout: float = 0.0,
        use_zero_inflation: bool = False,
    ):
        super().__init__()
        modules = []
        in_features = latent_dim
        for h in hidden_dims:
            modules += [
                nn.Linear(in_features, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_features = h
        self.encoder = nn.Sequential(*modules)

        self.skip = nn.Identity() if in_features == latent_dim else nn.Linear(latent_dim, in_features, bias=False)
        self.post_norm = nn.LayerNorm(in_features)

        # Mean parameter
        self.px_scale = nn.Linear(in_features, gene_dim)

        self.l_encoder = nn.Linear(in_features, 1)

        # Gene‑specific inverse dispersion (log‑space, broadcasted)
        self.log_theta = nn.Parameter(torch.randn(gene_dim))

        # Optional zero‑inflation gate
        self.use_zero_inflation = use_zero_inflation
        if use_zero_inflation:
            self.px_dropout = nn.Linear(in_features, gene_dim)

    @property
    def theta(self):
        # softplus to keep positive
        return F.softplus(self.log_theta)

    def forward(self, z: torch.Tensor, log_library: torch.Tensor | None = None):
        """
        z:            [B, latent_dim]
        log_library:  [B, 1]           (optional – if None we predict it)
        returns μ, θ (and π if requested)
        """
        flat = False
        if z.dim() == 3:  # [B,S,D]  → flatten
            B, S, D = z.shape
            z = z.reshape(-1, D)
            flat = True

        h = self.encoder(z)  # [B* S, H]
        h = self.post_norm(h + self.skip(z))

        if log_library is None:
            log_library = self.l_encoder(h)  # [B* S, 1]
        px_scale = F.softplus(self.px_scale(h))  # [B* S, G]
        mu = torch.exp(log_library) * px_scale  # NB mean

        if self.use_zero_inflation:
            pi = torch.sigmoid(self.px_dropout(h))
            outs = (mu, self.theta, pi)
        else:
            outs = (mu, self.theta)

        if flat:  # reshape back to [B,S,*]
            mu = mu.reshape(B, S, -1)
            if self.use_zero_inflation:
                pi = pi.reshape(B, S, -1)
                return mu, self.theta, pi  # θ remains [G]
            else:
                return mu, self.theta
        return outs

    def gene_dim(self) -> int:
        return self.px_scale.out_features


def nb_nll(x, mu, theta, eps: float = 1e-6):
    """
    Negative‑binomial negative log‑likelihood.
        x, mu : [..., G]
        theta : [G] or [..., G]
    returns scalar
    """
    logits = (mu + eps).log() - (theta + eps).log()  # NB parameterisation
    dist = NegativeBinomial(total_count=theta, logits=logits)
    return -dist.log_prob(x).mean()
