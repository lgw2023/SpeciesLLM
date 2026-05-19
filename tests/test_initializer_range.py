import importlib.util
import sys
import types
from pathlib import Path

import torch


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


def test_initializer_range_controls_linear_and_embedding_std():
    module = _load_adbc_model_module()
    torch.manual_seed(123)
    initializer_range = 0.003
    config = module.BERTConfig(
        vocab_size=64,
        hidden_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=256,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        max_position_embeddings=16,
        initializer_range=initializer_range,
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
    )

    model = module.BERTForPreTraining(config)
    bert_only = module.BERTModel(config)

    linear_std = model.bert.h[0].attention.self.query.weight.std(unbiased=False).item()
    embedding_std = model.bert.gene_encoder.embedding.weight.std(unbiased=False).item()
    bert_only_std = bert_only.h[0].attention.self.query.weight.std(unbiased=False).item()

    assert abs(linear_std - initializer_range) < 3e-4
    assert abs(embedding_std - initializer_range) < 3e-4
    assert abs(bert_only_std - initializer_range) < 3e-4
