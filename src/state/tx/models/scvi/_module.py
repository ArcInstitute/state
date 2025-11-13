from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence as kl
from torch_scatter import scatter_mean
from torchmetrics.functional import pairwise_euclidean_distance, pearson_corrcoef, r2_score
from torchmetrics.functional.clustering import normalized_mutual_info_score

from ._base_modules import CountDecoder, VariationalEncoder
from ._dists import NegativeBinomial, ZeroInflatedNegativeBinomial


def knn_purity(data, labels, n_neighbors=15):
    """Computes KNN Purity for ``data`` given the labels.
    Parameters
    ----------
    data:
        torch tensor of data (n_samples, n_features)
    labels
        torch tensor of labels (n_samples,)
    n_neighbors: int
        Number of nearest neighbors.
    Returns
    -------
    score: float
        KNN purity score. A float between 0 and 1.
    """
    distances = pairwise_euclidean_distance(data)
    # sort each row in distances to get nearest neighbors

    _, indices = torch.topk(distances, k=n_neighbors + 1, dim=1, largest=False, sorted=True)
    indices = indices[:, 1:]  # remove self
    # neighbors_labels = np.vectorize(lambda i: labels[i])(indices)
    neighbors_labels = labels[indices]  # (n_samples, n_neighbors)

    # pre cell purity scores
    scores = ((neighbors_labels - labels.reshape(-1, 1)) == 0).float().mean(axis=1)  # (n_samples,)
    res = scatter_mean(scores, labels).mean()  # per category purity

    return res


class scVIModule(nn.Module):
    """
    scVI module using NegativeBinomial/Zero-Inflated NegativeBinomial Likelihood objectives

    Parameters
    ----------
        n_genes: int
        n_treatments: int
        covars_encoder: dict
            Dictionary of covariates with keys as each covariate name and values as
                number of unique values of the corresponding covariate
        n_latent: int
            Latent Dimension
        loss_ae: str
            Autoencoder loss (either "gauss" or "nb")
        doser_type: str
            # Type of doser network, either `mlp` or `linear`.
        autoencoder_width: int
        autoencoder_depth: int
        use_batch_norm: bool
        use_layer_norm: bool
        variational: bool
    """

    def __init__(
        self,
        n_genes: int,
        n_perts: int,
        n_cell_types: int,
        n_batches: int = 1,
        pert_embeddings: Optional[np.ndarray] = None,
        n_latent: int = 128,
        n_pert_latent: int = 64,
        n_cell_type_latent: int = 32,
        n_batch_latent: int = 16,
        recon_loss: str = "nb",
        n_hidden_encoder: int = 256,
        n_layers_encoder: int = 3,
        n_hidden_decoder: int = 256,
        n_layers_decoder: int = 3,
        use_batch_norm: str = "both",
        use_layer_norm: str = "none",
        dropout_rate_encoder: float = 0.0,
        dropout_rate_decoder: float = 0.0,
        seed: int = 0,
    ):
        super().__init__()

        torch.manual_seed(seed)
        np.random.seed(seed)

        recon_loss = recon_loss.lower()

        assert recon_loss in ["nb", "zinb"]

        self.n_genes = n_genes
        self.n_latent = n_latent
        self.n_cell_type_latent = n_cell_type_latent
        self.n_batch_latent = n_batch_latent
        self.n_pert_latent = n_pert_latent
        self.n_perts = n_perts
        self.recon_loss = recon_loss

        self.encoder = VariationalEncoder(
            n_genes,
            n_latent,
            var_activation=nn.Softplus(),
            n_hidden=n_hidden_encoder,
            n_layers=n_layers_encoder,
            use_batch_norm=use_batch_norm in ["both", "encoder"],
            use_layer_norm=use_layer_norm in ["both", "encoder"],
            dropout_rate=dropout_rate_encoder,
            activation_fn=nn.ReLU,
            return_dist=True,
        )

        n_input_decoder = n_latent + n_cell_type_latent + n_batch_latent + n_pert_latent

        # Decoder components
        if self.recon_loss in ["zinb", "nb"]:
            # setup the parameters of your generative model, as well as your inference model
            self.px_r = torch.nn.Parameter(torch.randn(self.n_genes))

            # decoder goes from n_latent-dimensional space to n_input-d data
            self.decoder = CountDecoder(
                n_input=n_input_decoder,
                n_output=n_genes,
                n_layers=n_layers_decoder,
                n_hidden=n_hidden_decoder,
                use_norm=(
                    "batch"
                    if use_batch_norm in ["both", "decoder"]
                    else "layer"
                    if use_layer_norm in ["both", "decoder"]
                    else "none"
                ),
            )

        elif recon_loss == "gauss":
            self.decoder = VariationalEncoder(
                n_input=n_input_decoder,
                n_output=n_genes,
                n_layers=n_layers_decoder,
                n_hidden=n_hidden_decoder,
                dropout_rate=dropout_rate_decoder,
                use_norm=(
                    "batch"
                    if use_batch_norm in ["both", "decoder"]
                    else "layer"
                    if use_layer_norm in ["both", "decoder"]
                    else "none"
                ),
                var_activation=nn.Softplus(),
            )

        else:
            raise Exception("Invalid Loss function for Autoencoder")

        # Embeddings
        if pert_embeddings is not None:
            self.pert_embeddings = nn.Embedding.from_pretrained(torch.FloatTensor(pert_embeddings), freeze=True)
            self.n_pert_latent = pert_embeddings.shape[1]
        else:
            self.pert_embeddings = nn.Embedding(n_perts, n_pert_latent)

        if n_batches > 1:
            self.batch_embeddings = nn.Embedding(n_batches, n_batch_latent)

        self.cell_type_embeddings = nn.Embedding(n_cell_types, n_cell_type_latent)

        self.metrics = {
            "pearson_r": pearson_corrcoef,
            "r2_score": r2_score,
            "nmi": normalized_mutual_info_score,
        }

    def forward(self, x_basal, perts, cell_types, batch_ids, n_samples: int = 1):
        enc_outputs = self.forward_encoder(
            x=x_basal,
            perts=perts,
            cell_types=cell_types,
            batch_ids=batch_ids,
            n_samples=n_samples,
        )

        dec_outputs = self.forward_decoder(
            z_basal=enc_outputs["z_basal"],
            z=enc_outputs["z"],
            library=enc_outputs["library"],
        )

        return enc_outputs, dec_outputs

    def forward_encoder(
        self,
        x,
        perts,
        cell_types,
        batch_ids: Optional[torch.Tensor] = None,
        n_samples: int = 1,
    ):
        ## TODO: remove unused
        # batch_size = x.shape[0]

        if self.recon_loss in ["nb", "zinb"]:
            # log the input to the variational distribution for numerical stability
            x_ = torch.log(1 + x)
            library = torch.log(x.sum(1)).unsqueeze(1)
        else:
            x_ = x
            library = None, None

        qz, z_basal = self.encoder(x_)

        if n_samples > 1:
            sampled_z = qz.sample((n_samples,))
            z_basal = self.encoder.z_transformation(sampled_z)
            if self.recon_loss in ["nb", "zinb"]:
                library = library.unsqueeze(0).expand((n_samples, library.size(0), library.size(1)))

        z_covs = self.cell_type_embeddings(cell_types.long())
        z_batch = self.batch_embeddings(batch_ids.long())
        z_pert = self.pert_embeddings(perts.long())

        z = torch.cat([z_basal, z_covs, z_batch, z_pert], dim=-1)
        z_corrected = torch.cat([z_basal, z_covs, z_pert], dim=-1)

        return dict(
            z=z,
            z_basal=z_basal,
            z_corrected=z_corrected,
            library=library,
            qz=qz,
        )

    def forward_decoder(
        self,
        z_basal,
        z,
        library,
    ):
        if self.recon_loss == "nb":
            px_scale, _, px_rate, px_dropout = self.decoder("gene", z, library)
            px_r = torch.exp(self.px_r)

            px = NegativeBinomial(mu=px_rate, theta=px_r, scale=px_scale)

        elif self.recon_loss == "zinb":
            px_scale, _, px_rate, px_dropout = self.decoder("gene", z, library)
            px_r = torch.exp(self.px_r)

            px = ZeroInflatedNegativeBinomial(
                mu=px_rate,
                theta=px_r,
                zi_logits=px_dropout,
                scale=px_scale,
            )

        else:
            px_mean, px_var, x_pred = self.decoder(z)

            px = Normal(loc=px_mean, scale=px_var.sqrt())

        pz = Normal(torch.zeros_like(z_basal), torch.ones_like(z_basal))
        return dict(px=px, pz=pz)

    def loss(self, x_pert, encoder_outputs, decoder_outputs):
        """Computes the reconstruction loss (AE) or the ELBO (VAE)"""
        px = decoder_outputs["px"]
        recon_loss = -px.log_prob(x_pert).sum(dim=-1).mean()

        qz = encoder_outputs["qz"]
        pz = decoder_outputs["pz"]

        kl_divergence_z = kl(qz, pz).sum(dim=1)
        kl_loss = kl_divergence_z.mean()

        return recon_loss, kl_loss

    def r2_metric(self, x_pert, x_basal, encoder_outputs, decoder_outputs):
        px = decoder_outputs["px"]
        if self.recon_loss == "gauss":
            x_pred_mean = px.loc

            x_pred_mean = torch.nan_to_num(x_pred_mean, nan=0, posinf=1e3, neginf=-1e3)

            r2_mean = torch.nan_to_num(self.metrics["r2_score"](x_pred_mean.mean(0), x_pert.mean(0)), nan=0.0).item()

            lfc_true = (x_pert - x_basal).mean(0)
            lfc_pred = (x_pred_mean - x_basal).mean(0)

            r2_lfc = torch.nan_to_num(self.metrics["pearson_r"](lfc_pred, lfc_true), nan=0.0).item()

        elif self.recon_loss in ["nb", "zinb"]:
            x_pert = torch.log(1 + x_pert)
            x_pred = px.mu
            x_pred = torch.log(1 + x_pred)
            x_basal = torch.log(1 + x_basal)

            x_pred = torch.nan_to_num(x_pred, nan=0, posinf=1e3, neginf=-1e3)

            r2_mean = torch.nan_to_num(self.metrics["r2_score"](x_pred.mean(0), x_pert.mean(0)), nan=0.0).item()

            lfc_true = (x_pert - x_basal).mean(0)
            lfc_pred = (x_pred - x_basal).mean(0)

            r2_lfc = torch.nan_to_num(self.metrics["pearson_r"](lfc_pred, lfc_true), nan=0.0).item()

        return r2_mean, r2_lfc

    def sample(self, x_basal, perts, cell_types, batch_ids, n_samples=1):
        """Computes knockout gene expression

        Parameters
        ----------
        tensors : dict
            dictionary of input tensors

        """
        _, decoder_outputs = self.forward(
            x_basal=x_basal,
            perts=perts,
            cell_types=cell_types,
            batch_ids=batch_ids,
            n_samples=n_samples,
        )

        px = decoder_outputs["px"]

        if self.recon_loss == "gauss":
            output_key = "loc"
        else:
            output_key = "mu"

        output = getattr(px, output_key)

        return output
