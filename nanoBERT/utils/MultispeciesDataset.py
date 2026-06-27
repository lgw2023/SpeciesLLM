from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from torch.utils.data import Dataset, DataLoader
from .data_sampler import SubsetBatchSampler

import torch
import numpy as np
from scanpy import AnnData
import scanpy as sc
import pyarrow.parquet as pq

META_COLUMNS = {
    "species": "meta_species",
    "dataset_id": "meta_dataset_id",
    "assay": "meta_assay",
    "tissue": "meta_tissue",
    "tech_sample": "meta_tech_sample",
    "soma_joinid": "meta_soma_joinid",
    "idx": "meta_idx",
    "source_file_id": "meta_source_file_id",
    "source_batch_id": "meta_source_batch_id",
}


def add_meta_from_series(row_pt, row):
    for column, key in META_COLUMNS.items():
        if column in row.index:
            row_pt[key] = row[column]


def add_meta_from_frame_row(row_pt, row):
    for column, key in META_COLUMNS.items():
        if column in row.columns:
            row_pt[key] = row[column].values[0]


class SeqDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data

    def __len__(self):
        return self.data["genes"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


class ParquetDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        row_pt = {}
        row_pt["values"] = row['X']
        row_pt["seqmethod_labels"] = row['assay']
        row_pt["batch_labels"] = row['tech_sample']
        row_pt["species_labels"] = row['species']
        row_pt["tissue_labels"] = row['tissue']
        row_pt["disease_labels"] = row['disease']
        row_pt["sex_labels"] = row['sex']
        row_pt["age_labels"] = row['development_stage']
        add_meta_from_series(row_pt, row)
        return row_pt


class ParquetDataset_cls(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        row_pt = {}
        row_pt["values"] = row['X']
        row_pt["seqmethod_labels"] = row['assay']
        row_pt["batch_labels"] = row['tech_sample']
        row_pt["species_labels"] = row['species']
        row_pt["tissue_labels"] = row['tissue']
        row_pt["disease_labels"] = row['disease']
        row_pt["sex_labels"] = row['sex']
        row_pt["age_labels"] = row['development_stage']
        row_pt["celltype_labels"] = row['cell_type']
        add_meta_from_series(row_pt, row)
        return row_pt


class LazyParquetDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe
        self.len = len(self.df)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        row = self.df.loc[idx].compute()
        row_pt = {}
        row_pt["values"] = row['X'].values[0]
        row_pt["seqmethod_labels"] = row['assay'].values[0]
        row_pt["batch_labels"] = row['tech_sample'].values[0]
        row_pt["species_labels"] = row['species'].values[0]
        row_pt["tissue_labels"] = row['tissue'].values[0]
        row_pt["disease_labels"] = row['disease'].values[0]
        row_pt["sex_labels"] = row['sex'].values[0]
        row_pt["age_labels"] = row['development_stage'].values[0]
        add_meta_from_frame_row(row_pt, row)
        return row_pt


class PreindexedParquetDataset(Dataset):
    def __init__(self, parquet_files):
        # 仅加载元数据和行索引
        self.metadata = [(f, pq.read_metadata(f).num_rows) for f in parquet_files]
        self.cumulative_rows = np.cumsum([r for _, r in self.metadata])
        self.files = [f for f, _ in self.metadata]

    def __len__(self):
        return self.cumulative_rows[-1]

    def __getitem__(self, idx):
        # 定位文件
        file_idx = np.searchsorted(self.cumulative_rows,
                                   idx,
                                   side='right')
        if file_idx > 0:
            local_idx = idx - self.cumulative_rows[file_idx - 1]
        else:
            local_idx = idx

        # 使用pyarrow直接读取单行
        with pq.ParquetFile(self.files[file_idx]) as pf:
            table = pf.read_row_groups(row_groups=[local_idx // pf.metadata.row_group(0).num_rows],
                columns=['X', 'assay', 'tech_sample', 'species', 'tissue', 'disease', 'sex', 'development_stage']
                # 只读所需列
                )
            row = table.slice(local_idx % pf.metadata.row_group(0).num_rows,
                              1).to_pandas().iloc[0]
        row_pt = {}
        row_pt["values"] = row['X']
        row_pt["seqmethod_labels"] = row['assay']
        row_pt["batch_labels"] = row['tech_sample']
        row_pt["species_labels"] = row['species']
        row_pt["tissue_labels"] = row['tissue']
        row_pt["disease_labels"] = row['disease']
        row_pt["sex_labels"] = row['sex']
        row_pt["age_labels"] = row['development_stage']
        return row_pt


def prepare_data(tokenized_train_data, tokenized_valid_data, batch_labels_train, batch_labels_valid, config,
        cell_types_train=None, cell_types_valid=None, tissue_labels_train=None, tissue_labels_valid=None,
        species_labels_train=None, species_labels_valid=None, seqmethod_labels_train=None, seqmethod_labels_valid=None,
        sex_labels_train=None, sex_labels_valid=None, age_labels_train=None, age_labels_valid=None,
        disease_labels_train=None, disease_labels_valid=None, sort_seq_batch=False, ) -> Tuple[Dict[str, torch.Tensor]]:
    assert config.task in ["annotation", "integration", "perturbation", "multiomic"]

    gene_ids_train, gene_ids_valid = (tokenized_train_data["genes"], tokenized_valid_data["genes"])
    values_train, values_valid = (tokenized_train_data["values"], tokenized_valid_data["values"])
    embeddings_train, embeddings_valid = (tokenized_train_data["embeddings"], tokenized_valid_data["embeddings"])

    batch_labels_train = torch.from_numpy(batch_labels_train).long()
    batch_labels_valid = torch.from_numpy(batch_labels_valid).long()

    tissue_labels_train = torch.from_numpy(tissue_labels_train).long()
    tissue_labels_valid = torch.from_numpy(tissue_labels_valid).long()

    species_labels_train = torch.from_numpy(species_labels_train).long()
    species_labels_valid = torch.from_numpy(species_labels_valid).long()

    seqmethod_labels_train = torch.from_numpy(seqmethod_labels_train).long()
    seqmethod_labels_valid = torch.from_numpy(seqmethod_labels_valid).long()

    if sex_labels_train is not None:
        sex_labels_train = torch.from_numpy(sex_labels_train).long()
        sex_labels_valid = torch.from_numpy(sex_labels_valid).long()

    if age_labels_train is not None:
        age_labels_train = torch.from_numpy(age_labels_train).long()
        age_labels_valid = torch.from_numpy(age_labels_valid).long()

    if disease_labels_train is not None:
        disease_labels_train = torch.from_numpy(disease_labels_train).long()
        disease_labels_valid = torch.from_numpy(disease_labels_valid).long()

    if config.task == "annotation":
        cell_types_train = torch.from_numpy(cell_types_train).long()
        cell_types_valid = torch.from_numpy(cell_types_valid).long()

    if config.task == "multiomic":
        mode_types_train, mode_types_valid = (
            tokenized_train_data["mode_types"].long(), tokenized_valid_data["mode_types"].long(),)

    if sort_seq_batch:
        train_sort_ids = np.argsort(batch_labels_train)
        gene_ids_train = gene_ids_train[train_sort_ids]
        values_train = values_train[train_sort_ids]
        embeddings_train = embeddings_train[train_sort_ids]
        batch_labels_train = batch_labels_train[train_sort_ids]
        tissue_labels_train = tissue_labels_train[train_sort_ids]
        species_labels_train = species_labels_train[train_sort_ids]
        seqmethod_labels_train = seqmethod_labels_train[train_sort_ids]
        if sex_labels_train is not None:
            sex_labels_train = sex_labels_train[train_sort_ids]
        if age_labels_train is not None:
            age_labels_train = age_labels_train[train_sort_ids]
        if disease_labels_train is not None:
            disease_labels_train = disease_labels_train[train_sort_ids]
        if config.task == "annotation":
            cell_types_train = cell_types_train[train_sort_ids]
        if config.task == "multiomic":
            mode_types_train = mode_types_train[train_sort_ids]

        valid_sort_ids = np.argsort(batch_labels_valid)
        gene_ids_valid = gene_ids_train[valid_sort_ids]
        values_valid = values_train[valid_sort_ids]
        embeddings_valid = embeddings_valid[valid_sort_ids]
        batch_labels_valid = batch_labels_train[valid_sort_ids]
        tissue_labels_valid = tissue_labels_train[valid_sort_ids]
        species_labels_valid = species_labels_train[valid_sort_ids]
        seqmethod_labels_valid = seqmethod_labels_valid[valid_sort_ids]
        if sex_labels_valid is not None:
            sex_labels_valid = sex_labels_valid[valid_sort_ids]
        if age_labels_valid is not None:
            age_labels_valid = age_labels_valid[valid_sort_ids]
        if disease_labels_valid is not None:
            disease_labels_valid = disease_labels_valid[valid_sort_ids]
        if config.task == "annotation":
            cell_types_valid = cell_types_train[valid_sort_ids]
        if config.task == "multiomic":
            mode_types_valid = mode_types_train[valid_sort_ids]

    train_data = {
        "gene_ids": gene_ids_train, "values": values_train, "embeddings": embeddings_train,
        "batch_labels": batch_labels_train, "tissue_labels": tissue_labels_train,
        "species_labels": species_labels_train, "seq_method_labels": seqmethod_labels_train,
        }
    valid_data = {
        "gene_ids": gene_ids_valid, "values": values_valid, "embeddings": embeddings_valid,
        "batch_labels": batch_labels_valid, "tissue_labels": tissue_labels_valid,
        "species_labels": species_labels_valid, "seq_method_labels": seqmethod_labels_valid,
        }
    if config.use_sex_labels:
        train_data["sex_labels"] = sex_labels_train
        valid_data["sex_labels"] = sex_labels_valid
    if config.use_age_labels:
        train_data["age_labels"] = age_labels_train
        valid_data["age_labels"] = age_labels_valid
    if config.use_disease_labels:
        train_data["disease_labels"] = disease_labels_train
        valid_data["disease_labels"] = disease_labels_valid
    if config.task == "annotation":
        train_data["cell_types"] = cell_types_train
        valid_data["cell_types"] = cell_types_valid
    if config.task == "integration":
        train_data["mode_types"] = mode_types_train
        valid_data["mode_types"] = mode_types_valid

    return train_data, valid_data


def prepare_dataloader(data_pt: Dict[str, torch.Tensor], batch_size: int, shuffle: bool = False,
        intra_domain_shuffle: bool = False, drop_last: bool = False, num_workers: int = 0,
        per_seq_batch_sample: bool = False, ) -> DataLoader:
    dataset = SeqDataset(data_pt)

    if per_seq_batch_sample:
        # find the indices of samples in each seq batch
        subsets = []
        batch_labels_array = data_pt["batch_labels"].numpy()
        for batch_label in np.unique(batch_labels_array):
            batch_indices = np.where(batch_labels_array == batch_label)[0].tolist()
            subsets.append(batch_indices)
        data_loader = DataLoader(dataset=dataset,
            batch_sampler=SubsetBatchSampler(subsets,
                batch_size,
                intra_subset_shuffle=intra_domain_shuffle,
                inter_subset_shuffle=shuffle,
                drop_last=drop_last, ),
            num_workers=num_workers,
            pin_memory=True, )
        return data_loader

    data_loader = DataLoader(dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True, )
    return data_loader
