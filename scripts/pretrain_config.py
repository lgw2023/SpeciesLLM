"""Strict model-config loading shared by pretraining launchers and checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


MODEL_CONFIG_FIELD_MAP = {
    "HIDDEN_SIZE": "hidden_size",
    "NUM_HIDDEN_LAYERS": "num_hidden_layers",
    "NUM_ATTENTION_HEADS": "num_attention_heads",
    "INTERMEDIATE_SIZE": "intermediate_size",
    "HIDDEN_ACT": "hidden_act",
    "HIDDEN_DROPOUT_PROB": "hidden_dropout_prob",
    "CELL_HIDDEN_SIZE": "cell_hidden_size",
    "ATTENTION_PROBS_DROPOUT_PROB": "attention_probs_dropout_prob",
    "TYPE_VOCAB_SIZE": "type_vocab_size",
    "INITIALIZER_RANGE": "initializer_range",
    "LAYER_NORM_EPS": "layer_norm_eps",
    "ATTN_IMPLEMENTATION": "_attn_implementation",
    "USE_BATCH_LABELS": "use_batch_labels",
    "NUM_BATCH_LABELS": "num_batch_labels",
    "USE_SPECIES_LABELS": "use_species_labels",
    "NUM_SPECIES_LABELS": "num_species_labels",
    "USE_TISSUE_LABELS": "use_tissue_labels",
    "NUM_TISSUE_LABELS": "num_tissue_labels",
    "USE_SEQMETHOD_LABELS": "use_seqmethod_labels",
    "NUM_SEQMETHOD_LABELS": "num_seqmethod_labels",
    "USE_DISEASE_LABELS": "use_disease_labels",
    "NUM_DISEASE_LABELS": "num_disease_labels",
    "USE_AGE_LABELS": "use_age_labels",
    "NUM_AGE_LABELS": "num_age_labels",
    "USE_SEX_LABELS": "use_sex_labels",
    "NUM_SEX_LABELS": "num_sex_labels",
    "CELL_EMB_STYLE": "cell_emb_style",
    "CHUNK_SIZE_FEED_FORWARD": "chunk_size_feed_forward",
    "EXPLICIT_ZERO_PROB": "explicit_zero_prob",
}

LABEL_LIMIT_FIELDS = {
    "assay": "num_seqmethod_labels",
    "tech_sample": "num_batch_labels",
    "species": "num_species_labels",
    "tissue": "num_tissue_labels",
    "disease": "num_disease_labels",
    "development_stage": "num_age_labels",
    "sex": "num_sex_labels",
}

DEFAULT_GENE_EMBEDDING_MODALITIES = ("esm2", "gene_desc", "dnaseq")
GENE_EMBEDDING_FILES = {
    "esm2": "2nd_run_macrogene_features_sum_esm2.npy",
    "gene_desc": "2nd_run_macrogene_features_sum_gene_desc.npy",
    "dnaseq": "2nd_run_macrogene_features_sum_dnaseq.npy",
}


def shell_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def parse_gene_embedding_modalities(value: Any | None) -> tuple[str, ...]:
    if value is None:
        return DEFAULT_GENE_EMBEDDING_MODALITIES
    if isinstance(value, str):
        raw_modalities = [item.strip() for item in value.split(",")]
    else:
        raw_modalities = [str(item).strip() for item in value]

    modalities = [item for item in raw_modalities if item]
    if not modalities:
        raise ValueError("gene_embedding_modalities must contain at least one modality")

    invalid = [item for item in modalities if item not in GENE_EMBEDDING_FILES]
    if invalid:
        allowed = ", ".join(GENE_EMBEDDING_FILES)
        raise ValueError(
            f"Unsupported gene embedding modality: {', '.join(invalid)}. "
            f"Allowed values: {allowed}"
        )

    seen = set()
    duplicates = []
    for item in modalities:
        if item in seen and item not in duplicates:
            duplicates.append(item)
        seen.add(item)
    if duplicates:
        raise ValueError(f"Duplicate gene embedding modalities: {', '.join(duplicates)}")

    return tuple(modalities)


def gene_embedding_files_for_modalities(modalities: Any | None) -> list[tuple[str, str]]:
    return [
        (modality, GENE_EMBEDDING_FILES[modality])
        for modality in parse_gene_embedding_modalities(modalities)
    ]


def load_model_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise SystemExit(f"Missing fixed model config JSON: {config_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise SystemExit(f"{config_path}: expected a JSON object")

    missing_seq = [
        key
        for key in ("vocab_size", "max_position_embeddings")
        if key not in config or config[key] is None
    ]
    if missing_seq:
        raise SystemExit(
            f"{config_path}: missing required sequence fields: {', '.join(missing_seq)}"
        )

    vocab_seq_len = int(config["vocab_size"]) - 1
    position_seq_len = int(config["max_position_embeddings"]) - 1
    if vocab_seq_len != position_seq_len:
        raise SystemExit(
            f"{config_path}: vocab_size-1 ({vocab_seq_len}) != "
            f"max_position_embeddings-1 ({position_seq_len})"
        )
    config["seq_len"] = position_seq_len

    for json_key in MODEL_CONFIG_FIELD_MAP.values():
        if json_key not in config or config[json_key] is None:
            raise SystemExit(f"{config_path}: missing required field: {json_key}")

    hidden_size = int(config["hidden_size"])
    num_attention_heads = int(config["num_attention_heads"])
    if num_attention_heads <= 0:
        raise SystemExit(f"{config_path}: num_attention_heads must be > 0")
    if hidden_size % num_attention_heads != 0:
        raise SystemExit(
            f"{config_path}: hidden_size ({hidden_size}) must be divisible by "
            f"num_attention_heads ({num_attention_heads})"
        )
    return config


def label_limits_from_config(config: dict[str, Any]) -> dict[str, int]:
    return {
        label_column: int(config[json_key])
        for label_column, json_key in LABEL_LIMIT_FIELDS.items()
    }
