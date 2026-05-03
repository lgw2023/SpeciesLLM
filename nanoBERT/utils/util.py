from typing import Dict, Iterable, List, Optional, Tuple, Union, Mapping
from typing_extensions import Self
from pathlib import Path
from .. import logger
import logging
import numpy as np
import pandas as pd
import torch
from anndata import AnnData


def random_mask_value(values: Union[torch.Tensor, np.ndarray], mask_ratio: float = 0.15,
        mask_value: int = -1, ) -> torch.Tensor:
    """
	Randomly mask a batch of data. This method has been discarded and randomly mask is implementing in model class.

	Args:
		values (array-like): A batch of tokenized data, with shape (batch_size, n_features)
		mask_value (int): The value to mask with, default to -1.

	Returns:
		torch.Tensor: A tensor of masked data.
	"""
    if isinstance(values,
                  torch.Tensor):
        values = values.clone().detach().cpu().numpy()
    else:
        values = values.copy()

    ##Adapt randomly mask here
    for i in range(len(values)):
        row = values[i]
        indices = np.indices(row.shape)[0]
        n_mask = int(len(indices) * mask_ratio)
        mask_idx = np.random.choice(indices,
                                    n_mask,
                                    replace=False)
        row[mask_idx] = mask_value

    return torch.from_numpy(values).float()


def load_pretrained(model: torch.nn.Module, pretrained_params: Mapping[str, torch.Tensor], strict: bool = False,
        prefix: Optional[List[str]] = None, remove_prefix: Optional[str] = None, all_grad: bool = False,
        verbose: bool = None, ) -> torch.nn.Module:
    """
	Load pretrained weights to the model.

    Args:
        model (torch.nn.Module): The model to load weights to.
        pretrained_params (Mapping[str, torch.Tensor]): The pretrained parameters.
        strict (bool): Whether to strictly enforce that the keys in :attr:`pretrained_params`
            match the keys returned by this module's :meth:`Module.state_dict`. Default to False.
        prefix (List[str]): The list of prefix strings to match with the keys in
            :attr:`pretrained_params`. The matched keys will be loaded. Default to None.

    Returns:
        torch.nn.Module: The model with pretrained weights.
	"""

    if prefix is not None and len(prefix) > 0:
        if isinstance(prefix,
                      str):
            prefix = [prefix]
        pretrained_params = {k: v for k, v in pretrained_params.items() if any(k.startswith(p) for p in prefix)}
    if remove_prefix is not None:
        len_prefix = len(remove_prefix)
        pretrained_params = {k[len_prefix:]: v for k, v in pretrained_params.items() if k.startswith(remove_prefix)}

    model_dict = model.state_dict()
    if strict:
        if verbose:
            for k, v in pretrained_params.items():
                logger.info(f"Loading parameter {k} with shape {v.shape}")
        model_dict.update(pretrained_params)
        model.load_state_dict(model_dict)
    else:
        if verbose:
            print("Loading parameters according to their shape!")
            for k, v in pretrained_params.items():
                if k in model_dict and v.shape == model_dict[k].shape:
                    print(f"Loading parameter {k} with shape {v.shape}")
                    logger.info(f"Loading parameter {k} with shape {v.shape}")
        pretrained_params = {k: v for k, v in pretrained_params.items() if
            k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_params)
        model.load_state_dict(model_dict)

    if not all_grad:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.norm.parameters():
            param.requires_grad = False

    return model


def add_file_handler(logger: logging.Logger, log_file_path: Path):
    """
	Add a file handler to the logger.
	"""
    h = logging.FileHandler(log_file_path)

    # format showing time, name function and message
    formatter = logging.Formatter("%(asctime)s-%(name)s-%(levelname)s-%(funcName)s: %(message)s",
        datefmt="%H:%M:%S", )
    h.setFormatter(formatter)
    h.setLevel(logger.level)
    logger.addHandler(h)


def eval_scib_metrics(adata: AnnData, batch_key: str = "batch_labels", label_key: str = "celltype_labels",
        notes: Optional[str] = None, ) -> Dict:
    import scib

    results = scib.metrics.metrics(adata,
        adata_int=adata,
        batch_key=batch_key,
        label_key=label_key,
        embed="X_scGPT",
        isolated_labels_asw_=False,
        silhouette_=True,
        hvg_score_=False,
        graph_conn_=True,
        pcr_=True,
        isolated_labels_f1_=False,
        trajectory_=False,
        nmi_=True,
        # use the clustering, bias to the best matching
        ari_=True,
        # use the clustering, bias to the best matching
        cell_cycle_=False,
        kBET_=False,
        # kBET return nan sometimes, need to examine
        ilisi_=False,
        clisi_=False, )
    if notes is not None:
        logger.info(f"{notes}")

    logger.info(f"{results}")

    result_dict = results[0].to_dict()
    logger.info("Biological Conservation Metrics: \n"
                f"ASW (cell-type): {result_dict['ASW_label']:.4f}, graph cLISI: {result_dict['cLISI']:.4f}, "
                f"isolated label silhouette: {result_dict['isolated_label_silhouette']:.4f}, \n"
                "Batch Effect Removal Metrics: \n"
                f"PCR_batch: {result_dict['PCR_batch']:.4f}, ASW (batch): {result_dict['ASW_label/batch']:.4f}, "
                f"graph connectivity: {result_dict['graph_conn']:.4f}, graph iLISI: {result_dict['iLISI']:.4f}")

    result_dict["avg_bio"] = np.mean([result_dict["NMI_cluster/label"], result_dict["ARI_cluster/label"],
        result_dict["ASW_label"], ])

    # remove nan value in result_dict
    result_dict = {k: v for k, v in result_dict.items() if not np.isnan(v)}

    return result_dict