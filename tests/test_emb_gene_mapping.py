import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from state.emb.inference import Inference


@pytest.mark.parametrize(
    "index, column, genes, expected_mapping",
    [
        (["TP53", "ENSG2", "ENSG3", "ENSG4"], "gene_name", ["TP53", "ACTB", "GAPDH", "MALAT1"], [0, 1, 2, 3]),
        (["ENSG1", "ENSG2", "ENSG3", "ENSG4"], "gene_name", ["TP53", "ACTB", "GAPDH", "MALAT1"], [0, 1, 2, 3]),
        (["TP53", "ENSG2", "ENSG3", "ENSG4"], "gene_symbols", ["MALAT1", "GAPDH", "ACTB", "TP53"], [3, 2, 1, 0]),
        (["TP53", "ENSG2", "ENSG3", "ENSG4"], "gene_name", ["TP53", "ACTB", "GAPDH", "unknown"], [0, 1, 2, -1]),
        (["TP53", "ACTB", "GAPDH", "MALAT1"], "gene_name", ["unknown"] * 4, [0, 1, 2, 3]),
    ],
)
def test_encode_adata_uses_detected_gene_names(tmp_path, monkeypatch, index, column, genes, expected_mapping):
    counts = np.array([[0.0, 1.0, 2.0, 0.0]], dtype=np.float32)
    adata = ad.AnnData(counts, var=pd.DataFrame({column: pd.Categorical(genes)}, index=index))
    input_path = tmp_path / "input.h5ad"
    adata.write_h5ad(input_path)
    cfg = OmegaConf.create(
        {
            "embeddings": {"current": "test", "test": {"num": 4}},
            "dataset": {"current": "test", "test": {}, "pad_length": 3, "P": 1, "N": 0, "S": 0},
            "model": {"batch_size": 1},
            "loss": {"name": "mse"},
        }
    )
    protein_embeds = {gene: torch.zeros(2) for gene in ("TP53", "ACTB", "GAPDH", "MALAT1")}
    inference = Inference(cfg=cfg, protein_embeds=protein_embeds)

    # Keep real h5ad reading, column detection, dataset and collator construction;
    # replace only the model encoding call. No checkpoint or accelerator needed.
    def encode(dataloader):
        dataset = dataloader.dataset
        np.testing.assert_array_equal(dataset.ds_emb_map["input"], expected_mapping)
        expected_mask = np.asarray(expected_mapping) != -1
        np.testing.assert_array_equal(dataset.valid_gene_index["input"], expected_mask)
        np.testing.assert_array_equal(dataloader.collate_fn.valid_gene_mask["input"], expected_mask)
        np.testing.assert_array_equal(dataset[0][0].numpy(), counts)
        yield np.zeros((1, 2), dtype=np.float32), None

    monkeypatch.setattr(inference, "encode", encode)
    assert inference.encode_adata(str(input_path)).shape == (1, 2)
