from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union
from .gene_tokenizer import GeneVocab
from ..model.nanoBERTmodel import BERTConfig

import numpy as np
import torch


@dataclass
class CustomCollate_3GeneEmb:
    config: BERTConfig = None
    do_mlm: bool = True
    mlm_probability: float = 0.15
    mask_value: int = -1
    keep_first_n_tokens: int = 1
    return_pt: bool = True
    append_cls: bool = True
    include_zero_gene: bool = True
    cls_token: str = "<cls>"
    mode_type: np.ndarray = None
    vocab_mode: GeneVocab = None
    genes: np.ndarray = None
    esm_embeddings: np.ndarray = None
    desc_embeddings: np.ndarray = None
    dna_embeddings: np.ndarray = None

    def __call__(self, batch_data: List[Dict[str, np.ndarray]]):
        genes = self.genes
        gene_esm_embeddings = self.esm_embeddings
        gene_desc_embeddings = self.desc_embeddings
        gene_dna_embeddings = self.dna_embeddings
        tokenized_batch_data = self._tokenize_batch(
            batch_data,
            genes,
            gene_esm_embeddings,
            gene_desc_embeddings,
            gene_dna_embeddings,
        )
        return tokenized_batch_data

    def _tokenize_batch(
            self,
            data: List[Dict[str, np.ndarray]],
            gene_ids: np.ndarray,
            gene_esm_embeddings: np.ndarray,
            gene_desc_embeddings: np.ndarray,
            gene_dna_embeddings: np.ndarray,
    ) -> Dict[str, torch.Tensor]:

        cls_id = self.config.vocab[self.cls_token]
        if self.mode_type is not None:
            cls_id_mode_type = self.vocab_mode[self.cls_token]

        if data[0]["values"].size != len(gene_ids):
            print(data[0]["values"].size)
            raise ValueError(
                f"Number of features in data does not match"
                f"number of gene_ids ({len(gene_ids)})."
            )
        if self.mode_type is not None and data[0]["values"].size != len(self.mode_type):
            raise ValueError(
                f"Number of features in data does not match"
                f"number of mod_type ({len(self.mode_type)})."
            )

        batch_size = len(data)
        seq_len = len(gene_ids) + (1 if self.append_cls else 0)
        esm_embedding_dim = gene_esm_embeddings.shape[1]
        desc_embedding_dim = gene_desc_embeddings.shape[1]
        dna_embedding_dim = gene_dna_embeddings.shape[1]

        # Pre-allocate big batch tensors
        genes_tensor = np.zeros((batch_size, seq_len), dtype=np.int64)
        values_tensor = np.zeros((batch_size, seq_len), dtype=np.float32)
        esm_tensor = np.zeros((batch_size, seq_len, esm_embedding_dim), dtype=np.float32)
        desc_tensor = np.zeros((batch_size, seq_len, desc_embedding_dim), dtype=np.float32)
        dna_tensor = np.zeros((batch_size, seq_len, dna_embedding_dim), dtype=np.float32)

        if self.mode_type is not None:
            mode_types_tensor = np.zeros((batch_size, seq_len), dtype=np.int64)

        # Prepare static embeddings
        static_genes = gene_ids
        static_esm = gene_esm_embeddings
        static_desc = gene_desc_embeddings
        static_dna = gene_dna_embeddings
        if self.mode_type is not None:
            static_mode_type = self.mode_type

        if self.append_cls:
            static_genes = np.concatenate([[cls_id], static_genes])
            static_esm = np.concatenate([np.zeros((1, esm_embedding_dim)), static_esm])
            static_desc = np.concatenate([np.zeros((1, desc_embedding_dim)), static_desc])
            static_dna = np.concatenate([np.zeros((1, dna_embedding_dim)), static_dna])
            if self.mode_type is not None:
                static_mode_type = np.concatenate([[cls_id_mode_type], static_mode_type])

        # Now batch process
        # Vectorize extracting "values" and labels
        batch_labels_list = []
        species_labels_list = []
        tissue_labels_list = []
        seqmethod_labels_list = []
        disease_labels_list = []
        sex_labels_list = []
        age_labels_list = []
        celltype_labels_list = []

        for i, row in enumerate(data):
            row_values = row["values"]

            # Add cls if needed
            if self.append_cls:
                row_values = np.concatenate([[0], row_values])

            # Write to batch tensor
            genes_tensor[i] = static_genes
            values_tensor[i] = row_values
            esm_tensor[i] = static_esm
            desc_tensor[i] = static_desc
            dna_tensor[i] = static_dna
            if self.mode_type is not None:
                mode_types_tensor[i] = static_mode_type

            # Collect labels
            if self.config.use_batch_labels:
                batch_labels_list.append(row["batch_labels"])
            if self.config.use_species_labels:
                species_labels_list.append(row["species_labels"])
            if self.config.use_tissue_labels:
                tissue_labels_list.append(row["tissue_labels"])
            if self.config.use_seqmethod_labels:
                seqmethod_labels_list.append(row["seqmethod_labels"])
            if self.config.use_disease_labels:
                disease_labels_list.append(row["disease_labels"])
            if self.config.use_sex_labels:
                sex_labels_list.append(row["sex_labels"])
            if self.config.use_age_labels:
                age_labels_list.append(row["age_labels"])
            if self.config.do_cls:
                celltype_labels_list.append(row["celltype_labels"])

        # === Post processing ===
        tokenized_data = {
            "genes": torch.from_numpy(genes_tensor),
            "esm_embeddings": torch.from_numpy(esm_tensor),
            "desc_embeddings": torch.from_numpy(desc_tensor),
            "dna_embeddings": torch.from_numpy(dna_tensor),
        }

        # MLM masking
        expressions = torch.from_numpy(values_tensor)
        tokenized_data["target_values"] = expressions
        if self.do_mlm:
            probability_matrix = torch.full(expressions.shape, self.mlm_probability)
            if self.keep_first_n_tokens > 0:
                probability_matrix[:, :self.keep_first_n_tokens] = 0
            mask = torch.bernoulli(probability_matrix).bool()
            mask = mask.to(expressions.device)
            masked_expressions = expressions.masked_fill(mask, self.mask_value)
        else:
            masked_expressions = expressions

        tokenized_data["values"] = masked_expressions

        # Add mode_type if needed
        if self.mode_type is not None:
            tokenized_data["mode_types"] = torch.from_numpy(mode_types_tensor)

        # Add labels if needed
        if self.config.use_batch_labels:
            tokenized_data["batch_labels"] = torch.tensor(batch_labels_list, dtype=torch.long)
        if self.config.use_species_labels:
            tokenized_data["species_labels"] = torch.tensor(species_labels_list, dtype=torch.long)
        if self.config.use_tissue_labels:
            tokenized_data["tissue_labels"] = torch.tensor(tissue_labels_list, dtype=torch.long)
        if self.config.use_seqmethod_labels:
            tokenized_data["seqmethod_labels"] = torch.tensor(seqmethod_labels_list, dtype=torch.long)
        if self.config.use_disease_labels:
            tokenized_data["disease_labels"] = torch.tensor(disease_labels_list, dtype=torch.long)
        if self.config.use_sex_labels:
            tokenized_data["sex_labels"] = torch.tensor(sex_labels_list, dtype=torch.long)
        if self.config.use_age_labels:
            tokenized_data["age_labels"] = torch.tensor(age_labels_list, dtype=torch.long)
        if self.config.do_cls:
            tokenized_data["celltype_labels"] = torch.tensor(celltype_labels_list, dtype=torch.long)

        return tokenized_data
