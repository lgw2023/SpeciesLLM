import json
import pickle
from pathlib import Path
from collections import Counter, OrderedDict
from typing import Dict, Iterable, List, Optional, Tuple, Union
from typing_extensions import Self

import torch
import pandas as pd
import numpy as np


class Vocab:
    """Minimal torchtext-free Vocab implementation used by GeneVocab."""

    def __init__(self, tokens: Optional[Union[List[str], Dict[str, int]]] = None) -> None:
        self.default_index_: Optional[int] = None
        self.itos_: List[str] = []
        self.stoi_: Dict[str, int] = {}
        if tokens is None:
            return
        if isinstance(tokens, dict):
            ordered_tokens = [token for token, _ in sorted(tokens.items(), key=lambda item: item[1])]
        else:
            ordered_tokens = list(tokens)
        for token in ordered_tokens:
            self.append_token(token)

    def __len__(self) -> int:
        return len(self.itos_)

    def __contains__(self, token: str) -> bool:
        return token in self.stoi_

    def __getitem__(self, token: str) -> int:
        if token in self.stoi_:
            return self.stoi_[token]
        if self.default_index_ is not None:
            return self.default_index_
        raise KeyError(token)

    def _rebuild_stoi(self) -> None:
        self.stoi_ = {token: idx for idx, token in enumerate(self.itos_)}

    def set_default_index(self, index: Optional[int]) -> None:
        self.default_index_ = index

    def get_default_index(self) -> Optional[int]:
        return self.default_index_

    def insert_token(self, token: str, index: int) -> None:
        if token in self.stoi_:
            raise RuntimeError(f"Token {token!r} already exists in vocab")
        if index < 0 or index > len(self.itos_):
            raise RuntimeError(f"Index {index} is out of range for vocab size {len(self.itos_)}")
        self.itos_.insert(index, token)
        self._rebuild_stoi()

    def append_token(self, token: str) -> None:
        if token in self.stoi_:
            raise RuntimeError(f"Token {token!r} already exists in vocab")
        self.stoi_[token] = len(self.itos_)
        self.itos_.append(token)

    def lookup_token(self, index: int) -> str:
        return self.itos_[index]

    def lookup_tokens(self, indices: List[int]) -> List[str]:
        return [self.lookup_token(index) for index in indices]

    def lookup_indices(self, tokens: List[str]) -> List[int]:
        return [self[token] for token in tokens]

    def get_stoi(self) -> Dict[str, int]:
        return dict(self.stoi_)

    def get_itos(self) -> List[str]:
        return list(self.itos_)

class GeneVocab(Vocab):
    def __init__(
        self,
        gene_list_or_vocab: Union[List[str], Vocab],
        specials: Optional[List[str]] = None,
        special_first: bool = True,
    ) -> None:
        if isinstance(gene_list_or_vocab, Vocab):
            if specials is not None:
                raise ValueError("receive non-empty specials when init from a Vocab object.")
            super().__init__(gene_list_or_vocab.get_itos())
            self.set_default_index(gene_list_or_vocab.get_default_index())
        elif isinstance(gene_list_or_vocab, list):
            tokens = self._build_tokens_from_iterator(
                gene_list_or_vocab,
                specials=specials,
                special_first=special_first,
            )
            super().__init__(tokens)
        else:
            raise ValueError("gene_list_or_vocab must be a list of gene names or a Vocab object.")

    @classmethod
    def from_file(cls, file_path: Union[Path, str]) -> Self:
        if isinstance(file_path, str):
            file_path = Path(file_path)
        if file_path.suffix == ".pkl":
            with file_path.open("rb") as f:
                vocab = pickle.load(f)
                return cls(vocab)
        if file_path.suffix == ".json":
            with file_path.open("r") as f:
                token2idx = json.load(f)
                return cls.from_dict(token2idx)
        raise ValueError(f"{file_path} is not a valid file type. Only .pkl and .json are supported.")

    @classmethod
    def from_dict(cls, token2idx: Dict[str, int]) -> Self:
        vocab = cls([])
        for token, index in sorted(token2idx.items(), key=lambda item: item[1]):
            vocab.insert_token(token, index)
        return vocab

    def _build_tokens_from_iterator(
        self,
        iterator: Iterable,
        min_freq: int = 1,
        specials: Optional[List[str]] = None,
        special_first: bool = True,
    ) -> List[str]:
        counter = Counter()
        counter.update(iterator)
        if specials is not None:
            for token in specials:
                counter.pop(token, None)
        sorted_by_freq_tuples = sorted(counter.items(), key=lambda item: item[0])
        sorted_by_freq_tuples.sort(key=lambda item: item[1], reverse=True)
        ordered_dict = OrderedDict((token, count) for token, count in sorted_by_freq_tuples if count >= min_freq)
        tokens = list(ordered_dict.keys())
        if specials is not None:
            specials_to_add = list(specials)
            if special_first:
                tokens = specials_to_add + tokens
            else:
                tokens = tokens + specials_to_add
        return tokens

    def save_json(self, file_path: Union[Path, str]) -> None:
        if isinstance(file_path, str):
            file_path = Path(file_path)
        with file_path.open("w") as f:
            json.dump(self.get_stoi(), f, indent=2)

def get_customized_gene_vocab() -> GeneVocab:
	"""
	Get the default gene vocabylary, consisting of gene symbols and ids.
	"""
	vocab_file = Path(__file__).parent / "customized_gene_vocab.json"
	if not vocab_file.exists():
		print(
			f"No existing default vocab, will build one and save to {vocab_file}"
			)
		return _build_default_gene_vocab(save_vocab_to=vocab_file)

	print(f"Loading gene vocabulary from {vocab_file}")
	return GeneVocab.from_file(vocab_file)

def _build_default_gene_vocab(
	download_source_to: str = "/tmp",
	save_vocab_to: Union[Path, str, None] = None,
	) -> GeneVocab:
	"""
	Build the default gene vocabulary from HGNC gene symbols.

	Args:
		download_source_to (str): Directory to download the source data.
		save_vocab_to (Path or str): Path to save the vocabulary. If None,
			the vocabulary will not be save. Default to None.
	"""
	gene_collection_file = (
	    Path(download_source_to) / "human.gene_name_symbol.from_genenames.org.tsv"
	)
	if not gene_collection_file.exists():
	    # download and save file from url
		url = (
			"https://www.genenames.org/cgi-bin/download/custom?col=gd_app_sym&"
			"col=md_ensembl_id&status=Approved&status=Entry%20Withdrawn&hgnc_dbtag"
			"=on&order_by=gd_app_sym_sort&format=text&submit=submit"
		)
		import requests

		r = requests.get(url)
		gene_collection_file.write_text(r.text)

	print(f"Building gene vocabulary from {gene_collection_file}")
	df = pd.read_csv(gene_collection_file, sep="\t")
	gene_list = df["Approved symbol"].dropna().unique().tolist()
	gene_vocab = GeneVocab(gene_list)  # no special tokens set in default vocab
	if save_vocab_to is not None:
		gene_vocab.save_json(Path(save_vocab_to))
	return gene_vocab

def tokenize(
	data: np.ndarray,
	gene_ids: np.ndarray,
	gene_embeddings: np.ndarray,
	vocab: Vocab,
	return_pt: bool = True,
	append_cls: bool = True,
	include_zero_gene: bool = True,
	cls_token: str = "<cls>",
	mode_type: np.ndarray = None,
	vocab_mode: Vocab = None,
	) -> List[Tuple[Union[torch.Tensor, np.ndarray]]]:
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
	cls_id = vocab[cls_token]
	if mode_type is not None:
		cls_id_mode_type = vocab_mode[cls_token]

	if data.shape[1] != len(gene_ids):
		raise ValueError(
			f"Number of features in data ({data.shape[1]}) does not match"
			f"number of gene_ids ({len(gene_ids)})."
			)
	if mode_type is not None and data.shape[1] != len(mode_type):
		raise ValueError(
			f"Number of features in data ({data.shape[1]}) does not match"
			f"number of mod_type ({len(mode_type)})."
			)

	gene_ids_list = []
	values_list = []
	embeddings_list = []
	mode_types_list = []
	for i in range(len(data)):
		row = data[i]
		mode_types = None
		if include_zero_gene:
			values = row
			embeddings = gene_embeddings
			genes = gene_ids
			if mode_type is not None:
				mode_types = mode_type
		else:
			idx = np.nonzero(row)[0]
			values = row[idx]
			embeddings = gene_embeddings[idx]
			genes = gene_ids[idx]
			if mode_type is not None:
				mode_types = mode_type[idx]
		if append_cls:
			genes = np.insert(genes, 0, cls_id)
			values = np.insert(values, 0, 0)
			embeddings = np.insert(embeddings, 0, 0, 0)
			if mode_type is not None:
				mode_types = np.insert(mode_types, 0, cls_id_mode_type)
		if return_pt:
			genes = torch.from_numpy(genes).long()
			values = torch.from_numpy(values).float()
			embeddings = torch.from_numpy(embeddings).float()
			if mode_type is not None:
				mode_types = torch.from_numpy(mode_types).long()
		gene_ids_list.append(genes)
		values_list.append(values)
		embeddings_list.append(embeddings)
		if mode_type is not None:
			mode_types_list.append(mode_types)

	tokenized_data = {
		"genes": torch.stack(gene_ids_list, dim = 0),
		"values": torch.stack(values_list, dim=0),
		"embeddings": torch.stack(embeddings_list, dim=0),
	}
	if mode_type is not None:
		tokenized_data["mode_types"] = torch.stack(mode_types_list, dim=0)
	return tokenized_data


def tokenize_batch(
	data: np.ndarray,
	gene_ids: np.ndarray,
	gene_embeddings: np.ndarray,
	vocab: Vocab,
	return_pt: bool = True,
	append_cls: bool = True,
	include_zero_gene: bool = True,
	cls_token: str = "<cls>",
	mode_type: np.ndarray = None,
	vocab_mode: Vocab = None,
	mask_ratio: float = 0.15,
	mask_value: int = 0,
	keep_first_n_tokens: int = 1,
	) -> List[Tuple[Union[torch.Tensor, np.ndarray]]]:
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
	cls_id = vocab[cls_token]
	if mode_type is not None:
		cls_id_mode_type = vocab_mode[cls_token]

	if data[0]["values"] != len(gene_ids):
		raise ValueError(
			f"Number of features in data does not match"
			f"number of gene_ids ({len(gene_ids)})."
			)
	if mode_type is not None and data[0]["values"] != len(mode_type):
		raise ValueError(
			f"Number of features in data does not match"
			f"number of mod_type ({len(mode_type)})."
			)

	gene_ids_list = []
	values_list = []
	embeddings_list = []
	masked_expressions_list = []
	mode_types_list = []
	for i in range(len(data)):
		row = data[i]["values"]
		mode_types = None
		if include_zero_gene:
			values = row
			embeddings = gene_embeddings
			genes = gene_ids
			if mode_type is not None:
				mode_types = mode_type
		else:
			idx = np.nonzero(row)[0]
			values = row[idx]
			embeddings = gene_embeddings[idx]
			genes = gene_ids[idx]
			if mode_type is not None:
				mode_types = mode_type[idx]
		if append_cls:
			genes = np.insert(genes, 0, cls_id)
			values = np.insert(values, 0, 0)
			embeddings = np.insert(embeddings, 0, 0, 0)
			if mode_type is not None:
				mode_types = np.insert(mode_types, 0, cls_id_mode_type)

		if return_pt:
			genes = torch.from_numpy(genes).long()
			values = torch.from_numpy(values).float()
			embeddings = torch.from_numpy(embeddings).float()
			if mode_type is not None:
				mode_types = torch.from_numpy(mode_types).long()

		probability_matrix = torch.full(values.shape, mask_ratio)
		if keep_first_n_tokens > 0:
			probability_matrix[:, :keep_first_n_tokens] = 0
		mask = torch.bernoulli(probability_matrix).bool()
		mask = mask.to(values.device)
		masked_expressions = values.masked_fill(mask, mask_value)

		gene_ids_list.append(genes)
		values_list.append(values)
		masked_expressions_list.append(masked_expressions)
		embeddings_list.append(embeddings)
		if mode_type is not None:
			mode_types_list.append(mode_types)

	tokenized_data = {
		"genes": torch.stack(gene_ids_list, dim = 0),
		"values": torch.stack(masked_expressions_list, dim=0),
		"embeddings": torch.stack(embeddings_list, dim=0),
		"target_values": torch.stack(values_list, dim=0),
	}
	if mode_type is not None:
		tokenized_data["mode_types"] = torch.stack(mode_types_list, dim=0)
	return tokenized_data
