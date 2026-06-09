import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.pretrain_config import (
    gene_embedding_files_for_modalities,
    parse_gene_embedding_modalities,
)


def _load_adbc_model_module():
    root = Path(__file__).resolve().parents[1]
    nano_pkg = types.ModuleType("nanoBERT")
    nano_pkg.__path__ = [str(root / "nanoBERT")]
    model_pkg = types.ModuleType("nanoBERT.model")
    model_pkg.__path__ = [str(root / "nanoBERT" / "model")]
    sys.modules["nanoBERT"] = nano_pkg
    sys.modules["nanoBERT.model"] = model_pkg

    module_path = root / "nanoBERT" / "model" / "nanoBERTmodel_cellmeta2_plusEncode_adbc.py"
    spec = importlib.util.spec_from_file_location(
        "nanoBERT.model.nanoBERTmodel_cellmeta2_plusEncode_adbc",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_gene_embedding_modalities_for_two_way_ablation():
    assert parse_gene_embedding_modalities("esm2,dnaseq") == ("esm2", "dnaseq")
    assert gene_embedding_files_for_modalities("esm2,dnaseq") == [
        ("esm2", "2nd_run_macrogene_features_sum_esm2.npy"),
        ("dnaseq", "2nd_run_macrogene_features_sum_dnaseq.npy"),
    ]


@pytest.mark.parametrize("modalities", ["esm2,bad", "esm2,esm2", ""])
def test_parse_gene_embedding_modalities_rejects_invalid_values(modalities):
    with pytest.raises(ValueError):
        parse_gene_embedding_modalities(modalities)


def test_two_way_ablation_forward_does_not_require_gene_desc_embeddings():
    module = _load_adbc_model_module()
    torch.manual_seed(123)
    seq_len = 3
    batch_size = 2
    config = module.BERTConfig(
        vocab_size=seq_len + 2,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=32,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        max_position_embeddings=seq_len + 1,
        initializer_range=0.02,
        use_batch_labels=False,
        use_species_labels=False,
        use_tissue_labels=False,
        use_seqmethod_labels=False,
        use_disease_labels=False,
        use_sex_labels=False,
        use_age_labels=False,
        do_mvc=False,
        do_cls=False,
        explicit_zero_prob=False,
        gene_embedding_modalities=("esm2", "dnaseq"),
    )
    model = module.BERTForPreTraining(config)

    esm = np.zeros((seq_len, 1280), dtype=np.float32)
    dna = np.zeros((seq_len, 2560), dtype=np.float32)
    model.set_static_gene_inputs(
        np.arange(1, seq_len + 1),
        esm,
        None,
        dna,
        cls_id=0,
        append_cls=True,
    )

    values = torch.ones((batch_size, seq_len + 1), dtype=torch.float32)
    outputs = model(values=values, MVC=False)

    assert model.bert.enhanced_fusion.linear.in_features == 3840
    assert outputs["model_output"].shape == (batch_size, seq_len + 1)
