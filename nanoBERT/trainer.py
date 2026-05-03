import os
import time
from . import logger
from utils import masked_mse_loss, masked_relative_error, criterion_neg_log_bernoulli
import numpy as np
from scanpy import AnnData
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

criterion_cls = nn.CrossEntropyLoss()


def train(model: nn.Module, data_loader: DataLoader, lr, optimizer, scaler, device, grad_clip, config, ctx, logger,
        epoch, num_epochs, task: str = "annotation", gradient_accumulation_steps: int = 0, MLM: bool = True,
        master_process: bool = False, wandb_log: bool = False, ddp: bool = False, ) -> None:
    """
    Train the model for one epoch.
    """
    import wandb

    model.train()
    total_loss, total_gep_loss, total_gepc_loss, total_cls_loss = 0.0, 0.0, 0.0, 0.0
    total_zero_log_prob, total_gepc_zero_log_prob = 0.0, 0.0
    start_time = time.time()

    num_batches = len(data_loader)
    for batch, batch_data in enumerate(data_loader):
        input_gene_ids = batch_data["genes"].to(device)
        input_gene_values = batch_data["values"].to(device)
        input_gene_embeddings = batch_data["embeddings"].to(device)
        target_values = batch_data["target_values"].to(device)
        if config.use_batch_labels:
            input_batch_labels = batch_data["batch_labels"].to(device)
        if config.use_species_labels:
            input_species_labels = batch_data["species_labels"].to(device)
        if config.use_tissue_labels:
            input_tissue_labels = batch_data["tissue_labels"].to(device)
        if config.use_seqmethod_labels:
            input_seqmethod_labels = batch_data["seqmethod_labels"].to(device)
        if config.use_disease_labels:
            input_disease_labels = batch_data["disease_labels"].to(device)
        if config.use_sex_labels:
            input_sex_labels = batch_data["sex_labels"].to(device)
        if config.use_age_labels:
            input_age_labels = batch_data["age_labels"].to(device)
        if task == "annotation":
            celltypes_labels = batch_data["celltype_labels"].to(device)

        if ddp:
            model.require_backward_grad_sync = ((batch + 1) % gradient_accumulation_steps == 0)

        with ctx:
            outputs = model(src=input_gene_ids,
                values=input_gene_values,
                embeddings=input_gene_embeddings,
                batch_labels=input_batch_labels if config.use_batch_labels else None,
                species_labels=input_species_labels if config.use_species_labels else None,
                tissue_labels=input_tissue_labels if config.use_tissue_labels else None,
                seqmethod_labels=input_seqmethod_labels if config.use_seqmethod_labels else None,
                disease_labels=input_disease_labels if config.use_disease_labels else None,
                sex_labels=input_sex_labels if config.use_sex_labels else None,
                age_labels=input_age_labels if config.use_age_labels else None,
                CLS=config.do_cls,
                MVC=config.do_mvc,
                output_hidden_states=True,
                output_attentions=False,
                # can not both turned on with _attn_implementation of sdpa
                )

            mask_positions = input_gene_values.eq(-1)
            # compute loss
            loss = 0.0
            loss_to_log = {}
            if MLM:
                loss_gep = masked_mse_loss(outputs["model_output"],
                                           target_values,
                                           mask_positions)
                loss += loss_gep
                loss_to_log.update({"train/GEP": loss_gep.item()})
            if MLM and config.explicit_zero_prob:
                loss_zero_prob = criterion_neg_log_bernoulli(outputs["model_zero_prob"],
                                                             target_values,
                                                             mask_positions)
                loss += loss_zero_prob
                loss_to_log.update({"train/nzlp": loss_zero_prob.item()})
            if "mvc_output" in outputs:
                loss_gepc = masked_mse_loss(outputs["mvc_output"],
                                            target_values,
                                            mask_positions)
                loss += loss_gepc
                loss_to_log.update({"train/GEPC": loss_gepc.item()})
            if "mvc_output" in outputs and config.explicit_zero_prob:
                loss_gepc_zero_prob = criterion_neg_log_bernoulli(outputs["mvc_zero_probs"],
                                                                  target_values,
                                                                  mask_positions)
                loss += loss_gepc_zero_prob
                loss_to_log.update({"train/GEPC_nzlp": loss_gepc_zero_prob.item()})
            if config.do_cls:
                loss_cls = criterion_cls(outputs["cls_output"],
                                         celltypes_labels)
                loss += loss_cls
                loss_to_log.update({"train/cls": loss_cls.item()})

                error_rate = 1 - (
                    (outputs["cls_output"].argmax(1) == celltypes_labels).sum().item()) / celltypes_labels.size(0)

            if gradient_accumulation_steps != 0:
                loss = loss / gradient_accumulation_steps

        # backward and optimization
        scaler.scale(loss).backward()
        if (batch + 1) % gradient_accumulation_steps == 0 or (batch + 1) == num_batches:
            if grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        if master_process:
            lossf = loss.item() * gradient_accumulation_steps
            print(f"[{batch + 1}/{num_batches}] batches, loss is: {lossf:.4f}")
        total_loss += loss.item()
        total_gep_loss += loss_gep.item() if MLM else 0.0
        total_gepc_loss += loss_gepc.item() if config.do_mvc else 0.0
        total_cls_loss += loss_cls.item() if config.do_cls else 0.0
        total_zero_log_prob += (loss_zero_prob.item() if config.explicit_zero_prob else 0.0)
        total_gepc_zero_log_prob += (loss_gepc_zero_prob.item() if config.do_mvc and config.explicit_zero_prob else 0.0)
        if wandb_log and master_process:
            wandb.log(loss_to_log)

    if master_process:
        total_loss = total_loss * gradient_accumulation_steps / num_batches
        total_gep_loss /= num_batches
        total_gepc_loss /= num_batches
        total_cls_loss /= num_batches
        total_zero_log_prob /= num_batches
        total_gepc_zero_log_prob /= num_batches
        s_per_epoch = time.time() - start_time
        logger.info(f"Training | Epoch [{epoch + 1}/{num_epochs}] | Learning rate is: {lr:05.5f} | Time: {s_per_epoch:.2f}s | "
                    f"Average loss is: {total_loss:.4f} | " + (
                        f"average gep loss is: {total_gep_loss:.4f} | " if MLM else "") + (
                        f"average gepc loss is : {total_gepc_loss:.4f} | " if config.do_mvc else "") + (
                        f"average gep zero log prob loss is: {total_zero_log_prob:.4f} | " if config.explicit_zero_prob else "") + (
                        f"average gepc zero log prob loss is: {total_gepc_zero_log_prob:.4f} | " if config.do_mvc and config.explicit_zero_prob else "") + (
                        f"average cls loss is {total_cls_loss:.4f} | " if config.do_cls else ""))
        print(f"Epoch [{epoch + 1}/{num_epochs}], average loss is: {total_loss:.4f} | Learning rate is: {lr}")
    total_loss = 0
    total_gep_loss = 0
    total_gepc_loss = 0
    total_zero_log_prob = 0
    total_gepc_zero_log_prob = 0
    total_cls_loss = 0


def evaluate(model: nn.Module, loader: DataLoader, device, config, epoch, logger, ctx, wandb_log: bool = False,
        task: str = "annotation", ):
    """
    Evaluate the model on the evaluation data. Run this only on the master process.
    """
    import wandb

    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        num_batches = len(loader)
        for batch_data in loader:
            input_gene_ids = batch_data["genes"].to(device)
            input_gene_values = batch_data["values"].to(device)
            input_gene_embeddings = batch_data["embeddings"].to(device)
            target_values = batch_data["target_values"].to(device)
            if config.use_batch_labels:
                input_batch_labels = batch_data["batch_labels"].to(device)
            if config.use_species_labels:
                input_species_labels = batch_data["species_labels"].to(device)
            if config.use_tissue_labels:
                input_tissue_labels = batch_data["tissue_labels"].to(device)
            if config.use_seqmethod_labels:
                input_seqmethod_labels = batch_data["seqmethod_labels"].to(device)
            if config.use_disease_labels:
                input_disease_labels = batch_data["disease_labels"].to(device)
            if config.use_sex_labels:
                input_sex_labels = batch_data["sex_labels"].to(device)
            if config.use_age_labels:
                input_age_labels = batch_data["age_labels"].to(device)
            if task == "annotation":
                celltypes_labels = batch_data["celltype_labels"].to(device)

            with ctx:
                outputs = model(src=input_gene_ids,
                    values=input_gene_values,
                    embeddings=input_gene_embeddings,
                    batch_labels=input_batch_labels if config.use_batch_labels else None,
                    species_labels=input_species_labels if config.use_species_labels else None,
                    tissue_labels=input_tissue_labels if config.use_tissue_labels else None,
                    seqmethod_labels=input_seqmethod_labels if config.use_seqmethod_labels else None,
                    disease_labels=input_disease_labels if config.use_disease_labels else None,
                    sex_labels=input_sex_labels if config.use_sex_labels else None,
                    age_labels=input_age_labels if config.use_age_labels else None,
                    CLS=config.do_cls,
                    MVC=config.do_mvc,
                    output_hidden_states=True,
                    output_attentions=False,
                    # can not both turned on with _attn_implementation of sdpa
                    )

                mask_positions = input_gene_values.eq(-1)
                if task == "annotation":
                    loss = criterion_cls(outputs["cls_output"],
                                         celltypes_labels)
                elif task in ["integration", "multiomic"]:
                    loss = masked_mse_loss(outputs["mlm_output"],
                                           target_values,
                                           mask_positions)

            total_loss += loss.item()
        total_loss /= num_batches
    print(f"Valid | Epoch [{epoch + 1}] | Average inference loss is: {total_loss:.4f} ")
    logger.info(f"Valid | Epoch [{epoch + 1}] | Average inference loss is: {total_loss:.4f} ")
    if wandb_log:
        wandb.log({
            "valid/loss": total_loss, "epoch": epoch + 1,
            })
    return total_loss


def predict(model: nn.Module, data_loader: DataLoader, config, device, ctx, task: str = "annotation", ) -> float:
    """
    Using the pre-trained model to predict cell types.
    """
    model.eval()

    predictions = []
    with torch.no_grad():
        for batch_data in data_loader:
            input_gene_ids = batch_data["genes"].to(device)
            input_gene_values = batch_data["values"].to(device)
            input_gene_embeddings = batch_data["embeddings"].to(device)
            if config.use_batch_labels:
                input_batch_labels = batch_data["batch_labels"].to(device)
            if config.use_species_labels:
                input_species_labels = batch_data["species_labels"].to(device)
            if config.use_tissue_labels:
                input_tissue_labels = batch_data["tissue_labels"].to(device)
            if config.use_seqmethod_labels:
                input_seqmethod_labels = batch_data["seqmethod_labels"].to(device)
            if config.use_disease_labels:
                input_disease_labels = batch_data["disease_labels"].to(device)
            if config.use_sex_labels:
                input_sex_labels = batch_data["sex_labels"].to(device)
            if config.use_age_labels:
                input_age_labels = batch_data["age_labels"].to(device)
            if task == "annotation":
                celltypes_labels = batch_data["celltype_labels"].to(device)

            with ctx:
                outputs = model(src=input_gene_ids,
                    values=input_gene_values,
                    embeddings=input_gene_embeddings,
                    batch_labels=input_batch_labels if config.use_batch_labels else None,
                    species_labels=input_species_labels if config.use_species_labels else None,
                    tissue_labels=input_tissue_labels if config.use_tissue_labels else None,
                    seqmethod_labels=input_seqmethod_labels if config.use_seqmethod_labels else None,
                    disease_labels=input_disease_labels if config.use_disease_labels else None,
                    sex_labels=input_sex_labels if config.use_sex_labels else None,
                    age_labels=input_age_labels if config.use_age_labels else None,
                    CLS=config.do_cls,
                    MVC=config.do_mvc,
                    output_hidden_states=True,
                    output_attentions=False,
                    # can not both turned on with _attn_implementation of sdpa
                    )
                predict = outputs["cls_output"]
                preds = predict.argmax(1).cpu().numpy()
                predictions.append(preds)
    return np.concatenate(predictions,
                          axis=0)


def inference(model: nn.Module, adata: AnnData, vocab, config, device, logger, wandb_log: bool = False, ) -> float:
    """
    Inference API for cell types predictions.
    """
    # TODO: processing adata

    return None