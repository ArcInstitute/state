from typing import Literal, Optional

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn import functional as F


class FocalLoss(nn.Module):
    """Inspired by https://github.com/AdeelH/pytorch-multi-class-focal-loss/blob/master/focal_loss.py

    Focal Loss, as described in https://arxiv.org/abs/1708.02002.
    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    x is expected to contain raw, unnormalized scores for each class.
    y is expected to contain class labels.
    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        """
        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper.
                Defaults to 0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
        """
        if reduction not in ("mean", "sum", "none"):
            raise ValueError('Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

        self.nll_loss = nn.NLLLoss(weight=alpha, reduction="none")

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if len(y_true) == 0:
            return torch.tensor(0.0)

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(y_pred, dim=-1)
        ce = self.nll_loss(log_p, y_true)

        # get true class column from each row
        all_rows = torch.arange(len(y_pred))
        log_pt = log_p[all_rows, y_true]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt) ** self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


class MLP(nn.Module):
    def __init__(
        self,
        n_input,
        n_output,
        n_hidden,
        n_layers,
        activation_fn: Optional[nn.Module] = nn.ReLU,
        use_norm: str = "batch",
        dropout_rate: float = 0.3,
        drop_norm_last_layer: bool = True,
    ):
        super().__init__()
        if drop_norm_last_layer:
            layers = [n_input] + [n_hidden] * n_layers
        else:
            layers = [n_input] + [n_hidden] * (n_layers - 1) + [n_output]

        network = []
        for n_in, n_out in zip(layers[:-1], layers[1:]):
            network.append(nn.Linear(n_in, n_out))
            if use_norm == "batch":
                network.append(nn.BatchNorm1d(n_out))
            elif use_norm == "layer":
                network.append(nn.LayerNorm(n_out))
            network.append(activation_fn())
            network.append(nn.Dropout(dropout_rate))

        if drop_norm_last_layer:
            network.append(nn.Linear(n_hidden, n_output))

        self.network = nn.Sequential(*network)

    def forward(self, x):
        """
        x: (batch_size, n_input)
        """
        return self.network(x)


class Classifier(nn.Module):
    def __init__(
        self,
        n_input,
        n_labels,
        n_hidden,
        n_layers,
        activation_fn=nn.ReLU,
        use_norm: str = "batch",
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        self.n_output = n_labels

        self.network = MLP(
            n_input=n_input,
            n_output=n_labels,
            n_layers=n_layers,
            n_hidden=n_hidden,
            use_norm=use_norm,
            dropout_rate=dropout_rate,
            activation_fn=activation_fn,
            drop_norm_last_layer=True,
        )

    def forward(self, x):
        y = self.network(x)
        return y


class VariationalEncoder(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        use_norm: str = "batch",
        var_eps: float = 1e-4,
        var_activation=None,
        return_dist: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.var_eps = var_eps
        self.encoder = MLP(
            n_input=n_input,
            n_output=n_hidden,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_norm=use_norm,
            drop_norm_last_layer=False,
        )
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)
        self.return_dist = return_dist

        self.var_activation = torch.exp if var_activation is None else var_activation

    def forward(self, x: torch.Tensor, *cat_list: int):
        """ """
        q = self.encoder(x, *cat_list)

        q_m = self.mean_encoder(q)
        q_v = self.var_activation(self.var_encoder(q)) + self.var_eps

        dist = Normal(q_m, q_v.sqrt())
        latent = dist.rsample()

        if self.return_dist:
            return dist, latent

        return q_m, q_v, latent


# Inspired by scvi-tools source code: https://github.com/scverse/scvi-tools/blob/d094c9b3c14e8cb3ac3a309b9cf0160aff237393/scvi/nn/_base_components.py
class CountDecoder(nn.Module):
    """Decodes data from latent space of ``n_input`` dimensions into ``n_output`` dimensions.

    Uses a fully-connected neural network of ``n_hidden`` layers.

    Parameters
    ----------
    n_input
        The dimensionality of the input (latent space)
    n_output
        The dimensionality of the output (data space)
    n_cat_list
        A list containing the number of categories
        for each category of interest. Each category will be
        included using a one-hot encoding
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    inject_covariates
        Whether to inject covariates in each layer, or just the first (default).
    use_batch_norm
        Whether to use batch norm in layers
    use_layer_norm
        Whether to use layer norm in layers
    scale_activation
        Activation layer to use for px_scale_decoder
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_layers: int = 1,
        n_hidden: int = 128,
        use_norm: Literal["batch", "layer"] = "batch",
        scale_activation: Literal["softmax", "softplus"] = "softmax",
    ):
        super().__init__()
        self.px_decoder = MLP(
            n_input=n_input,
            n_output=n_hidden,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0.0,
            use_norm=use_norm,
            drop_norm_last_layer=False,
        )

        # mean gamma
        if scale_activation == "softmax":
            px_scale_activation = nn.Softmax(dim=-1)
        elif scale_activation == "softplus":
            px_scale_activation = nn.Softplus()
        self.px_scale_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_output),
            px_scale_activation,
        )

        # dispersion: here we only deal with gene-cell dispersion case
        self.px_r_decoder = nn.Linear(n_hidden, n_output)

        # dropout
        self.px_dropout_decoder = nn.Linear(n_hidden, n_output)

    def forward(
        self,
        dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"],
        z: torch.Tensor,
        library: torch.Tensor,
    ):
        """The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns parameters for the ZINB distribution of expression
         #. If ``dispersion != 'gene-cell'`` then value for that param will be ``None``

        Parameters
        ----------
        z :
            tensor with shape ``(n_input,)``
        library_size
            library size
        cat_list
            list of category membership(s) for this sample
        dispersion
            One of the following

            * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
            * ``'gene-batch'`` - dispersion can differ between different batches
            * ``'gene-label'`` - dispersion can differ between different labels
            * ``'gene-cell'`` - dispersion can differ for every gene in every cell

        Returns
        -------
        4-tuple of :py:class:`torch.Tensor`
            parameters for the ZINB distribution of expression

        """
        # The decoder returns values for the parameters of the ZINB distribution
        px = self.px_decoder(z)
        px_scale = self.px_scale_decoder(px)
        px_dropout = self.px_dropout_decoder(px)
        # Clamp to high value: exp(12) ~ 160000 to avoid nans (computational stability)
        px_rate = torch.exp(library) * px_scale  # torch.clamp( , max=12)
        px_r = self.px_r_decoder(px) if dispersion == "gene-cell" else None
        return px_scale, px_r, px_rate, px_dropout


class GeneralizedSigmoid(nn.Module):
    """
    Sigmoid, log-sigmoid or linear functions for encoding dose-response for
    drug perurbations.
    """

    def __init__(self, n_drugs, non_linearity="sigmoid"):
        """Sigmoid modeling of continuous variable.
        Params
        ------
        nonlin : str (default: logsigm)
            One of logsigm, sigm.
        """
        super(GeneralizedSigmoid, self).__init__()
        self.non_linearity = non_linearity
        self.n_drugs = n_drugs

        self.beta = torch.nn.Parameter(torch.ones(1, n_drugs), requires_grad=True)
        self.bias = torch.nn.Parameter(torch.zeros(1, n_drugs), requires_grad=True)

        self.vmap = None

    def forward(self, x, y):
        """
        Parameters
        ----------
        x: (batch_size, max_comb_len)
        y: (batch_size, max_comb_len)
        """
        y = y.long()
        if self.non_linearity == "logsigm":
            bias = self.bias[0][y]
            beta = self.beta[0][y]
            c0 = bias.sigmoid()
            return (torch.log1p(x) * beta + bias).sigmoid() - c0
        elif self.non_linearity == "sigm":
            bias = self.bias[0][y]
            beta = self.beta[0][y]
            c0 = bias.sigmoid()
            return (x * beta + bias).sigmoid() - c0
        else:
            return x
