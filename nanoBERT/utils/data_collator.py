from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union
from .gene_tokenizer import GeneVocab
from ..model.nanoBERTmodel import BERTConfig

import numpy as np
import torch


@dataclass
class CustomCollate:
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
    embeddings: np.ndarray = None

    def __call__(self, batch_data: List[Dict[str, np.ndarray]]):
        genes = self.genes
        gene_embeddings = self.embeddings
        tokenized_batch_data = self._tokenize_batch(batch_data,
            genes,
            gene_embeddings, )
        return tokenized_batch_data

    def _tokenize_batch(self, data: np.ndarray, gene_ids: np.ndarray, gene_embeddings: np.ndarray, ) -> List[
        Tuple[Union[torch.Tensor, np.ndarray]]]:
        """
        Tokenize a batch of data. Returns a list of tuple (gene_id, count).

        Args:
            data (array-like): A batch of data, with shape (batch_size, n_features).
                n_features equals the number of all genes.
            gene_ids (array-like): A batch of gene ids, with shape (n_features,).
            return_pt (bool): Whether to return torch tensors of gene_ids and counts,
                default to True.

        Returns:
            list: A list of tuple (gene_id, count) of gene expressions.
        """
        cls_id = self.config.vocab[self.cls_token]
        if self.mode_type is not None:
            cls_id_mode_type = self.vocab_mode[self.cls_token]

        if data[0]["values"].size != len(gene_ids):
            print(data[0]["values"].size)
            raise ValueError(f"Number of features in data does not match"
                             f"number of gene_ids ({len(gene_ids)}).")
        if self.mode_type is not None and data[0]["values"].size != len(self.mode_type):
            raise ValueError(f"Number of features in data does not match"
                             f"number of mod_type ({len(self.mode_type)}).")

        gene_ids_list = []
        values_list = []
        embeddings_list = []
        mode_types_list = []
        batch_list = []
        species_list = []
        tissue_list = []
        seqmethod_list = []
        disease_list = []
        sex_list = []
        age_list = []
        for i in range(len(data)):
            row = data[i]["values"]
            mode_types = None
            if self.include_zero_gene:
                values = row
                embeddings = gene_embeddings
                genes = gene_ids
                if self.mode_type is not None:
                    mode_types = self.mode_type
            else:
                idx = np.nonzero(row)[0]
                values = row[idx]
                embeddings = gene_embeddings[idx]
                genes = gene_ids[idx]
                if self.mode_type is not None:
                    mode_types = self.mode_type[idx]
            if self.append_cls:
                genes = np.insert(genes,
                                  0,
                                  cls_id)
                values = np.insert(values,
                                   0,
                                   0)
                embeddings = np.insert(embeddings,
                                       0,
                                       0,
                                       0)
                if self.mode_type is not None:
                    mode_types = np.insert(mode_types,
                                           0,
                                           cls_id_mode_type)

            if self.return_pt:
                genes = torch.from_numpy(genes).long()
                values = torch.from_numpy(values).float()
                embeddings = torch.from_numpy(embeddings).float()
                if self.mode_type is not None:
                    mode_types = torch.from_numpy(mode_types).long()

            gene_ids_list.append(genes)
            values_list.append(values)
            embeddings_list.append(embeddings)
            if self.mode_type is not None:
                mode_types_list.append(mode_types)
            if self.config.use_batch_labels:
                batch_list.append(data[i]["batch_labels"])
            if self.config.use_species_labels:
                species_list.append(data[i]["species_labels"])
            if self.config.use_tissue_labels:
                tissue_list.append(data[i]["tissue_labels"])
            if self.config.use_seqmethod_labels:
                seqmethod_list.append(data[i]["seqmethod_labels"])
            if self.config.use_disease_labels:
                disease_list.append(data[i]["disease_labels"])
            if self.config.use_sex_labels:
                sex_list.append(data[i]["sex_labels"])
            if self.config.use_age_labels:
                age_list.append(data[i]["age_labels"])

        expressions = torch.stack(values_list,
                                  dim=0)
        tokenized_data = {
            "genes": torch.stack(gene_ids_list,
                                 dim=0), "embeddings": torch.stack(embeddings_list,
                                                                   dim=0), "target_values": expressions,
            }
        if self.do_mlm:
            probability_matrix = torch.full(expressions.shape,
                                            self.mlm_probability)
            if self.keep_first_n_tokens > 0:
                probability_matrix[:, :self.keep_first_n_tokens] = 0
            mask = torch.bernoulli(probability_matrix).bool()
            mask = mask.to(expressions.device)
            masked_expressions = expressions.masked_fill(mask,
                                                         self.mask_value)
        else:
            masked_expressions = expressions
        tokenized_data["values"] = masked_expressions

        if self.mode_type is not None:
            tokenized_data["mode_types"] = torch.stack(mode_types_list,
                                                       dim=0)
        if self.config.use_batch_labels:
            tokenized_data["batch_labels"] = torch.tensor(batch_list,
                                                          dtype=torch.long)
        if self.config.use_species_labels:
            tokenized_data["species_labels"] = torch.tensor(species_list,
                                                            dtype=torch.long)
        if self.config.use_tissue_labels:
            tokenized_data["tissue_labels"] = torch.tensor(tissue_list,
                                                           dtype=torch.long)
        if self.config.use_seqmethod_labels:
            tokenized_data["seqmethod_labels"] = torch.tensor(seqmethod_list,
                                                              dtype=torch.long)
        if self.config.use_disease_labels:
            tokenized_data["disease_labels"] = torch.tensor(disease_list,
                                                            dtype=torch.long)
        if self.config.use_sex_labels:
            tokenized_data["sex_labels"] = torch.tensor(sex_list,
                                                        dtype=torch.long)
        if self.config.use_age_labels:
            tokenized_data["age_labels"] = torch.tensor(age_list,
                                                        dtype=torch.long)
        return tokenized_data
