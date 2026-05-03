import argparse
import json
import os
import glob
import h5py
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Union, Optional
from pathlib import Path
from tqdm import tqdm
import scanpy as sc
import dask

dask.config.set({"dataframe.convert-string": False})
import dask.dataframe as dd
from dask import delayed

from nanoBERT.utils import CustomCollate_3GeneEmb, ParquetDataset, GeneVocab, load_pretrained
from nanoBERT.model.nanoBERTmodel_cellmeta2_plusEncode_adbc import BERTForPreTraining, BERTConfig

PathLike = Union[str, os.PathLike]


def read_parquet(fp):
    return delayed(pd.read_parquet)(fp)


def load_data_for_rank(folder_path, rank, world_size):
    file_paths = sorted(glob.glob(f"{folder_path}/*.parquet"))
    if not file_paths:
        raise FileNotFoundError(f"No parquet files found in {folder_path}")

    sub_file_paths = file_paths[rank::world_size]
    print(f"[Rank {rank}] will read files: {sub_file_paths}",
          flush=True)

    delayed_reads = [read_parquet(fp) for fp in sub_file_paths]
    ddf = dd.from_delayed(delayed_reads)
    pdf = ddf.compute()
    print(f"[Rank {rank}] total rows read: {len(pdf)}",
          flush=True)

    return pdf


def load_data(folder_path):
    if os.path.isdir(folder_path):
        file_paths = sorted(glob.glob(f"{folder_path}/*.parquet"))
        if not file_paths:
            raise FileNotFoundError(f"No parquet files found in {folder_path}")
        delayed_reads = [read_parquet(fp) for fp in file_paths]
    elif os.path.isfile(folder_path):
        delayed_reads = read_parquet(folder_path)
    else:
        raise ValueError(f"Data path {folder_path} is neither a directory or a file!")

    ddf = dd.from_delayed(delayed_reads)
    pdf = ddf.compute()
    print(f"Total rows read: {len(pdf)}",
          flush=True)

    return pdf


def get_cell_gene_embeddings_from_data(data_pt: pd.DataFrame, model: torch.nn.Module = None,
        cell_embedding_mode: str = "cls", collate_fn=None, batch_size=16, seq_len=None, gene_ids=None,
        get_gene_embeddings=False, use_metadata=True, ) -> torch.Tensor:
    cell_num = data_pt["X"].shape[0]
    gene_num = seq_len
    dataloader = DataLoader(dataset=ParquetDataset(data_pt),
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=min(len(os.sched_getaffinity(0)),
                        batch_size // 2),
        pin_memory=True, )
    if cell_embedding_mode == "cls":
        device = next(model.parameters()).device
        cell_embeddings = []
        gene_embeddings = []
        emb_num = 0

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
            for batch_data in tqdm(dataloader,
                                   desc="Embedding cells"):
                input_gene_ids = batch_data["genes"].to(device)
                input_gene_values = batch_data["values"].to(device)
                input_gene_embeddings = batch_data["esm_embeddings"].to(device)
                input_desc_embeddings = batch_data["desc_embeddings"].to(device)
                input_dna_embeddings = batch_data["dna_embeddings"].to(device)
                target_values = batch_data["target_values"].to(device)

                if use_metadata:
                    if "batch_labels" in batch_data:
                        input_batch_labels = batch_data["batch_labels"].to(device)
                    else:
                        input_batch_labels = None
                    if "species_labels" in batch_data:
                        input_species_labels = batch_data["species_labels"].to(device)
                    else:
                        input_species_labels = None
                    if "tissue_labels" in batch_data:
                        input_tissue_labels = batch_data["tissue_labels"].to(device)
                    else:
                        input_tissue_labels = None
                    if "seqmethod_labels" in batch_data:
                        input_seqmethod_labels = batch_data["seqmethod_labels"].to(device)
                    else:
                        input_seqmethod_labels = None
                    if "disease_labels" in batch_data:
                        input_disease_labels = batch_data["disease_labels"].to(device)
                    else:
                        input_disease_labels = None
                    if "sex_labels" in batch_data:
                        input_sex_labels = batch_data["sex_labels"].to(device)
                    else:
                        input_sex_labels = None
                    if "age_labels" in batch_data:
                        input_age_labels = batch_data["age_labels"].to(device)
                    else:
                        input_age_labels = None
                    if "celltype_labels" in batch_data:
                        input_celltype_labels = batch_data["celltype_labels"].to(device)
                    else:
                        input_celltype_labels = None
                else:
                    input_batch_labels = None
                    input_species_labels = None
                    input_tissue_labels = None
                    input_seqmethod_labels = None
                    input_disease_labels = None
                    input_sex_labels = None
                    input_age_labels = None
                    input_celltype_labels = None

                outputs, cell_emb, bert_hidden_states, bert_attentions = model._encode(src=input_gene_ids,
                    values=input_gene_values,
                    esm_embeddings=input_gene_embeddings,
                    desc_embeddings=input_desc_embeddings,
                    dna_embeddings=input_dna_embeddings,
                    batch_labels=input_batch_labels,
                    species_labels=input_species_labels,
                    tissue_labels=input_tissue_labels,
                    seqmethod_labels=input_seqmethod_labels,
                    disease_labels=input_disease_labels,
                    sex_labels=input_sex_labels,
                    age_labels=input_age_labels,
                    output_hidden_states=True,
                    output_attentions=False,
                    # can not both turned on with _attn_implementation of sdpa
                    )
                cell_batch_embs = cell_emb.cpu().numpy()
                gene_batch_embs = outputs[:, 1:, :].cpu().numpy()
                cell_embeddings.append(cell_batch_embs)
                gene_embeddings.append(gene_batch_embs)
                gene_batch_embs = None
                cell_batch_embs = None

            # emb_num = cell_embeddings[0].shape[1]
            # print("Start to stack multi-batch embeddings!")
            # cell_embeddings_total = np.empty((cell_num, emb_num), dtype = cell_embeddings[0].dtype)
            # gene_embeddings_total = np.empty((cell_num, gene_num, emb_num), dtype = cell_embeddings[0].dtype)
            # start_idx = 0
            # for (batch_cell_embs, batch_gene_embs) in zip(cell_embeddings, gene_embeddings):
            #     batch_size_cells = batch_cell_embs.shape[0]
            #     cell_embeddings_total[start_idx : start_idx + batch_size_cells] = batch_cell_embs
            #     gene_embeddings_total[start_idx : start_idx + batch_size_cells] = batch_gene_embs
            #     start_idx += batch_size_cells

            cell_embeddings = np.concatenate(cell_embeddings,
                                             axis=0)
            if get_gene_embeddings:
                gene_embeddings = np.concatenate(gene_embeddings,
                                                 axis=0)
                if gene_ids is not None:
                    gene_embeddings = gene_embeddings[:, gene_ids, :]
            else:
                gene_embeddings = None
    else:
        raise ValueError(f"Unknown cell embedding mode: {cell_embedding_mode}")

    return cell_embeddings, gene_embeddings


def embed_data(pdf: Union[pd.DataFrame, PathLike], esm_embeddings: np.ndarray, desc_embeddings: np.ndarray,
        dna_embeddings: np.ndarray, model_path: PathLike, is_pretrained_model: bool = True,
        remove_prefix: Optional[str] = None, seq_len: int = None, batch_size=32, obs_to_save: Optional[list] = None,
        gene_ids: Optional[list] = None, device: Union[str, torch.device] = "cuda", use_fast_transformer: bool = True,
        use_metadata: bool = True, get_gene_embeddings: bool = False, return_adata: bool = False, ) -> Optional[
    np.ndarray | sc.AnnData]:
    if isinstance(obs_to_save,
                  str):
        assert obs_to_save in pdf.columns, f"obs_to_save {obs_to_save} not in data's columns"
        obs_to_save = [obs_to_save]

    if gene_ids is not None:
        for gene_id in gene_ids:
            if gene_id > seq_len:
                raise ValueError(f"Gene id {gene_id} is exceed the max len of sequence!")

    if device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            print("WARNING: CUDA is not available, using CPU instead!")

    vocab_file = "customized_gene" + str(seq_len) + "_vocab.json"
    model_config_file = "args_ft_batch.json"
    special_tokens = ["<cls>", "<mask>", "<unk>", "<sep>"]

    vocab = GeneVocab.from_file(vocab_file)
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)

    with open(model_config_file,
              "r") as f:
        model_configs = json.load(f)
    if use_fast_transformer:
        print("Using fast mode")
        model_configs["_attn_implementation"] = "sdpa"

    config = BERTConfig(**model_configs)
    config.vocab = vocab

    src = np.arange(1,
                    seq_len + 1)
    collate_fn = CustomCollate_3GeneEmb(config=config,
                                        do_mlm=False,
                                        genes=src,
                                        esm_embeddings=esm_embeddings,
                                        desc_embeddings=desc_embeddings,
                                        dna_embeddings=dna_embeddings)

    model = BERTForPreTraining(config)
    if is_pretrained_model:
        load_pretrained(model,
                        torch.load(model_path,
                                   map_location=device)['model'],
                        remove_prefix=remove_prefix,
                        all_grad=True,
                        verbose=False)
    else:
        load_pretrained(model,
                        torch.load(model_path,
                                   map_location=device),
                        remove_prefix=remove_prefix,
                        all_grad=True,
                        verbose=False)
    model.to(device)
    model.eval()

    cell_embeddings, gene_embeddings = get_cell_gene_embeddings_from_data(pdf,
        model,
        "cls",
        collate_fn,
        batch_size,
        seq_len,
        gene_ids,
        get_gene_embeddings,
        use_metadata, )

    if return_adata:
        values = np.vstack(pdf["X"])
        adata = sc.AnnData(X=values)
        adata.obs = pdf[obs_to_save]
        adata.obsm["X_SpeciesLLM"] = cell_embeddings
        return adata

    return (cell_embeddings, gene_embeddings) if get_gene_embeddings else cell_embeddings


if __name__ == "__main__":
    data_path = "/ibex/scratch/projects/c2307/codes/SpeciesLLM/test_data_inte/"
    emb_path = "/ibex/scratch/projects/c2307/macrogene"
    model_path = "out/best_model_epoch_1.pth"
    seq_len = 862
    batch_size = 1
    esm_embeddings = np.load(emb_path + "/1st_run_macrogene_features_sum_esm2.npy")
    desc_embeddings = np.load(emb_path + "/1st_run_macrogene_features_sum_gene_desc.npy")
    dna_embeddings = np.load(emb_path + "/1st_run_macrogene_features_sum_dnaseq.npy")
    file_name = "human_macrogene_1234.parquet"
    data_file = data_path + file_name

    return_adata = True
    remove_prefix = None
    is_pretrained_model = False
    get_gene_embeddings = False

    if isinstance(data_file,
                  pd.DataFrame):
        pdf = data_file
    elif isinstance(data_file,
                    str):
        pdf = load_data(data_file)

    unique_batch = pdf["tech_sample"].unique()
    batch_to_index = {batch: i for i, batch in enumerate(unique_batch)}
    pdf["tech_sample"] = pdf["tech_sample"].map(batch_to_index)

    if return_adata:
        obs_to_save = ["cell_type", "tech_sample"]
        adata = embed_data(pdf,
            esm_embeddings,
            desc_embeddings,
            dna_embeddings,
            model_path,
            is_pretrained_model,
            remove_prefix,
            seq_len,
            batch_size,
            obs_to_save,
            get_gene_embeddings=get_gene_embeddings,
            return_adata=return_adata, )
        adata.write_h5ad(data_file.split(".")[0] + "_SpeciesLLM_epoch1.h5ad",
                         compression="gzip")

    else:
        obs_to_save = None
        cell_gene_embs = embed_data(pdf,
            esm_embeddings,
            desc_embeddings,
            dna_embeddings,
            model_path,
            load_pretrained,
            seq_len,
            batch_size,
            get_gene_embeddings=get_gene_embeddings,
            return_adata=return_adata, )

        save_file_name = file_name.split(".")[0]
        gene_embs_file = "cell_embs/" + save_file_name + "_gene_embs.h5"
        cell_embs_file = "cell_embs/" + save_file_name + "_cell_embs.npy"

        # with h5py.File(gene_embs_file, "w") as f:
        #     f.create_dataset("matrix", data = gene_embs, compression = "gzip")

        np.save(cell_embs_file,
                cell_gene_embs[0])



