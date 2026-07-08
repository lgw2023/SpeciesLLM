import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.pretrain_config import (
    gene_embedding_files_for_modalities,
    load_model_config,
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


def test_gene_embedding_suffix_selects_v2_embedding_files():
    assert gene_embedding_files_for_modalities("esm2,gene_desc,dnaseq", suffix="_v2") == [
        ("esm2", "2nd_run_macrogene_features_sum_esm2_v2.npy"),
        ("gene_desc", "2nd_run_macrogene_features_sum_gene_desc_v2.npy"),
        ("dnaseq", "2nd_run_macrogene_features_sum_dnaseq_v2.npy"),
    ]


@pytest.mark.parametrize("modalities", ["esm2,bad", "esm2,esm2", ""])
def test_parse_gene_embedding_modalities_rejects_invalid_values(modalities):
    with pytest.raises(ValueError):
        parse_gene_embedding_modalities(modalities)


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        (
            "args_2nd_run_100m_v2_604.json",
            {
                "seq_len": 604,
                "hidden_size": 640,
                "num_hidden_layers": 12,
                "num_attention_heads": 10,
                "intermediate_size": 5120,
                "num_species_labels": 49,
            },
        ),
        (
            "args_2nd_run_500m_v2_604.json",
            {
                "seq_len": 604,
                "hidden_size": 1280,
                "num_hidden_layers": 24,
                "num_attention_heads": 20,
                "intermediate_size": 5120,
                "num_species_labels": 49,
            },
        ),
        (
            "args_2nd_run_1b_v2_604.json",
            {
                "seq_len": 604,
                "hidden_size": 1440,
                "num_hidden_layers": 40,
                "num_attention_heads": 30,
                "intermediate_size": 5760,
                "num_species_labels": 49,
            },
        ),
    ],
)
def test_v2_604_model_configs_are_strictly_loadable(filename, expected):
    config_path = Path(__file__).resolve().parents[1] / "Stage2_macrogene_embeddings" / filename
    config = load_model_config(config_path)

    for key, value in expected.items():
        assert config[key] == value
    assert config["vocab_size"] == 605
    assert config["max_position_embeddings"] == 605
    assert config["hidden_size"] % config["num_attention_heads"] == 0


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


def test_two_way_ablation_train_losses_use_all_trainable_parameters():
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
        do_mvc=True,
        do_cls=False,
        explicit_zero_prob=True,
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
    values[:, 1] = -1.0
    target = torch.zeros_like(values)
    mask = values.eq(-1).float()
    outputs = model(values=values, MVC=True)

    zero_prob = outputs["model_zero_prob"].clamp(1e-6, 1 - 1e-6)
    mvc_zero_prob = outputs["mvc_zero_probs"].clamp(1e-6, 1 - 1e-6)
    loss = ((outputs["model_output"] - target).pow(2) * mask).sum()
    loss = loss - (zero_prob.log() * mask).sum()
    loss = loss + ((outputs["mvc_output"] - target).pow(2) * mask).sum()
    loss = loss - (mvc_zero_prob.log() * mask).sum()
    loss.backward()

    parameter_names = dict(model.named_parameters())
    unused = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    assert "bert.enhanced_fusion.norm_desc.weight" not in parameter_names
    assert "bert.enhanced_fusion.norm_desc.bias" not in parameter_names
    assert unused == []
