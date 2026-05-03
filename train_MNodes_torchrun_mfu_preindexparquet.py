import os
import sys
import dask
import json
import time
import math
import glob
import logging
import argparse
import datetime
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import moxing as mox
from copy import deepcopy
from dataclasses import dataclass
from contextlib import nullcontext

import torch
import torch._dynamo
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

import apex
import torch_npu
from torch_npu.contrib import transfer_to_npu
from torch_npu.npu.amp import GradScaler, autocast

from nanoBERT.utils import GeneVocab, CustomCollate_3GeneEmb
from nanoBERT.utils import ParquetDataset, LazyParquetDataset, PreindexedParquetDataset
from nanoBERT.utils import masked_mse_loss, masked_relative_error, criterion_neg_log_bernoulli
from nanoBERT.model.nanoBERTmodel_cellmeta2_plusEncode_adbc import BERTConfig, BERTForPreTraining

import dask
import dask.dataframe as dd
from dask import delayed
dask.config.set({"dataframe.convert-string": False})
torch._dynamo.config.suppress_errors = True

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class DistributedFileSampler:
    def __init__(self, file_paths, num_replicas=None, rank=None, shuffle=True, seed=0, drop_last=False):
        self.file_paths = file_paths
        self.num_samples = len(file_paths)
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last

        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0

        if self.drop_last and self.num_samples % self.num_replicas != 0:
            self.num_samples_per_rank = int(math.ceil((self.num_samples - self.num_replicas) / self.num_replicas))
        else:
            self.num_samples_per_rank = int(math.ceil(self.num_samples / self.num_replicas))
        self.total_size = self.num_samples_per_rank * self.num_replicas

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        # 1. Shuffle all indices with epoch-based seed
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = list(torch.randperm(self.num_samples, generator=g).tolist())
        else:
            indices = list(range(self.num_samples))

        if not self.drop_last:
            # 2. Pad to total_size if needed
            padding_size = self.total_size - len(indices)
            if padding_size < len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]

        else:
            indices = indices[:self.total_size]

        assert len(indices) == self.total_size

        # 3. Subsample for this rank
        # This is chatgpt's suggestion
        # offset = self.num_samples_per_rank * self.rank
        # indices = indices[offset:offset + self.num_samples_per_rank]

        # This is DistributedFileSampler's solution
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples_per_rank

        return iter(indices)

    def __len__(self):
        return self.num_samples_per_rank

def get_lr(itertion_number, min_learning_rate, warmup_iters, learning_rate, lr_decay_iters):
    assert min_learning_rate <= learning_rate
    # (1) linear warmup for warmup_epoches
    if itertion_number < warmup_iters:
        return learning_rate * (itertion_number + 1) / (warmup_iters + 1)
    if itertion_number > lr_decay_iters:
        return min_learning_rate
    decay_ratio = (itertion_number - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_learning_rate + coeff * (learning_rate - min_learning_rate)


def read_parquet(fp):
    return dask.delayed(pd.read_parquet)(fp)


def read_parquet_delayed(fp):
    return dask.dataframe.read_parquet(fp, engine="pyarrow", memory_map=True)


def load_data_for_rank(folder_path, rank, world_size):
    file_paths = sorted(glob.glob(f"{folder_path}/*.parquet"))
    if not file_paths:
        raise FileNotFoundError(f"No parquet files found in {folder_path}")

    sub_file_paths = file_paths[rank::world_size]
    print(f"[Rank {rank}] will read files: {sub_file_paths}", flush=True)

    delayed_reads = [read_parquet(fp) for fp in sub_file_paths]
    ddf = dask.dataframe.from_delayed(delayed_reads)
    pdf = ddf.compute()
    print(f"[Rank {rank}] total rows read: {len(pdf)}", flush=True)

    return pdf


def load_data_for_total(folder_path):
    file_paths = sorted(glob.glob(f"{folder_path}/*.parquet"))
    if not file_paths:
        raise FileNotFoundError(f"No parquet files found in {folder_path}")

    delayed_reads = [read_parquet(fp) for fp in file_paths]
    ddf = dask.dataframe.from_delayed(delayed_reads)
    pdf = ddf.compute()

    return pdf


def load_data(folder_path):
    file_paths = sorted(glob.glob(f"{folder_path}/*.parquet"))
    if not file_paths:
        raise FileNotFoundError(f"No parquet files found in {folder_path}")

    delayed_reads = read_parquet_delayed(file_paths)

    return delayed_reads


def load_data_for_rank_with_DistributedFileSampler(train_data_filelist, file_indices, rank, NODE_RANK):
    file_paths = deepcopy(train_data_filelist)
    if not train_data_filelist:
        raise FileNotFoundError(f"No parquet files found in {train_data_filelist}")

    sub_file_paths = [file_paths[i] for i in file_indices]
    print(f"[Node {NODE_RANK} Rank {rank}] will read files: {[i.split('/')[-1] for i in sub_file_paths]}", flush=True)

    delayed_reads = [read_parquet(fp) for fp in sub_file_paths]
    ddf = dd.from_delayed(delayed_reads)
    pdf = ddf.compute()
    print(f"[Node {NODE_RANK} Rank {rank}] total rows read: {len(pdf)}", flush=True)

    return pdf, sub_file_paths


def get_files(folder_path, number=0):
    if number == 0:
        file_paths = sorted(glob.glob(f"{folder_path}/*.parquet"))
    else:
        file_paths = sorted(glob.glob(f"{folder_path}/*.parquet"))[:number]
    if not file_paths:
        raise FileNotFoundError(f"No parquet files found in {folder_path}")
    return file_paths


def build_bertconfig(vocab, seq_len, args):
    # load config
    config = BERTConfig(vocab_size=seq_len + 1,
                        vocab=vocab,
                        hidden_size=args.hidden_size if args.hidden_size else 1280,
                        num_hidden_layers=args.num_hidden_layers if args.num_hidden_layers else 24,
                        num_attention_heads=args.num_attention_heads if args.num_attention_heads else 20,
                        intermediate_size=args.intermediate_size if args.intermediate_size else 5120,
                        hidden_act=args.hidden_act if args.hidden_act else "gelu",
                        hidden_dropout_prob=args.hidden_dropout_prob if args.hidden_dropout_prob else 0.1,
                        cell_hidden_size=args.cell_hidden_size if args.cell_hidden_size else 128,
                        attention_probs_dropout_prob=args.attention_probs_dropout_prob if args.attention_probs_dropout_prob else 0.1,
                        max_position_embeddings=args.seq_len + 1,
                        type_vocab_size=args.type_vocab_size if args.type_vocab_size else 2,
                        initializer_range=args.initializer_range if args.initializer_range else 0.02,
                        layer_norm_eps=args.layer_norm_eps if args.layer_norm_eps else 1e-12,
                        _attn_implementation=args._attn_implementation if args._attn_implementation else "sdpa",
                        use_batch_labels=str2bool(args.use_batch_labels) if args.use_batch_labels is not None else False,
                        num_batch_labels=args.num_batch_labels if args.num_batch_labels else 12028,
                        use_species_labels=str2bool(args.use_species_labels) if args.use_species_labels is not None else True,
                        num_species_labels=args.num_species_labels if args.num_species_labels else 11,
                        use_tissue_labels=str2bool(args.use_tissue_labels) if args.use_tissue_labels is not None else True,
                        num_tissue_labels=args.num_tissue_labels if args.num_tissue_labels else 154,
                        use_seqmethod_labels=str2bool(args.use_seqmethod_labels) if args.use_seqmethod_labels is not None else True,
                        num_seqmethod_labels=args.num_seqmethod_labels if args.num_seqmethod_labels else 28,
                        use_disease_labels=str2bool(args.use_disease_labels) if args.use_disease_labels is not None else True,
                        num_disease_labels=args.num_disease_labels if args.num_disease_labels else 143,
                        use_age_labels=str2bool(args.use_age_labels) if args.use_age_labels is not None else True,
                        num_age_labels=args.num_age_labels if args.num_age_labels else 5,
                        use_sex_labels=str2bool(args.use_sex_labels) if args.use_sex_labels is not None else True,
                        num_sex_labels=args.num_sex_labels if args.num_sex_labels else 3,
                        cell_emb_style=args.cell_emb_style if args.cell_emb_style else "cls",
                        chunk_size_feed_forward=args.chunk_size_feed_forward if args.chunk_size_feed_forward is not None else 0,
                        explicit_zero_prob=str2bool(args.explicit_zero_prob) if args.explicit_zero_prob is not None else True,
                        )

    return config


def setup_ddp(backend='hccl', device_type='npu'):
    ddp = int(os.environ.get('RANK', -1)) != -1
    if True:
        torch.distributed.init_process_group(backend=backend)
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch_npu.npu.set_device(local_rank)
        master_process = (rank == 0 and local_rank == 0)
        world_size = int(os.environ['WORLD_SIZE']) if int(os.environ['WORLD_SIZE']) else int(os.environ['NNODES']) * int(os.environ['NPROC_PER_NODE'])
    else:
        local_rank = 0
        rank = 0
        master_process = True
        world_size = 1

    return ddp, rank, local_rank, master_process, world_size


def train_loop(args, model, ddp, rank, optimizer, train_data_filelist, train_sampler, device, config, ctx, scaler, grad_clip, out_dir, collate_fn, world_size):
    raw_model = model.module if ddp else model
    iter_num, runing_mfu = 0, -1.0

    save_step_interval = int(args.save_data_interval / world_size / args.batch_size)
    print(f"save_step_interval {save_step_interval} = save_data_interval {args.save_data_interval} / world_size {world_size} / batch_size {args.batch_size}")
    NODE_RANK = os.getenv('NODE_RANK') if os.getenv('NODE_RANK') else os.getenv('node_rank')
    local_log_file = os.path.join(out_dir, f"log.{NODE_RANK}-{rank}.txt")
    loss_to_log = {}

    t_temp_1_sum_lr = 0
    t_temp_2_sum_data = 0
    t_temp_3_sum_batchdata = 0
    t_temp_4_sum_sync = 0
    t_temp_5_sum_forward = 0
    t_temp_6_sum_loss_gep = 0
    t_temp_7_sum_loss_gep_zero_prob = 0
    t_temp_8_sum_backward = 0
    t_temp_9_sum_step_update_grad = 0
    t_temp_10_sum_sync = 0

    t_temp_0 = time.time()
    for epoch in range(args.epoch):

        # Uncomment this if use DistributedSampler
        t_temp_2_data = time.time()
        if ddp:
            train_sampler.set_epoch(epoch)
            data_pt, file_paths = load_data_for_rank_with_DistributedFileSampler(
                train_data_filelist,
                train_sampler,
                rank,
                NODE_RANK)
            print(f"Node: {NODE_RANK}, Rank: {rank}, Epoch: {epoch}, Data: {[x.split('/')[-1] for x in file_paths]}")
            write_to_log(f"Node: {NODE_RANK}, Rank: {rank}, Epoch: {epoch}, Data: {[x.split('/')[-1] for x in file_paths]}", local_log_file)
            data_loader = DataLoader(dataset=ParquetDataset(data_pt),
                batch_size=args.batch_size,
                collate_fn=collate_fn,
                shuffle=False,
                drop_last=False,
                num_workers=min(len(os.sched_getaffinity(0)), args.batch_size // 2),
                pin_memory=True)
        t_temp_2_data = time.time() - t_temp_2_data
        t_temp_2_sum_data += t_temp_2_data

        model.train()
        total_loss = 0.0
        num_batches = len(data_loader) # 已经处理过 // args.batch_size

        # 确保 local_step 是一个 tensor，并且在 GPU 上或 CPU 上与进程环境一致
        step_tensor = torch.tensor([num_batches], dtype=torch.long).to("npu" if torch.cuda.is_available() else "cpu")
        # 使用 all_reduce 进行全局最小值操作
        dist.all_reduce(step_tensor, op=dist.ReduceOp.MIN)
        # 返回全局最小值
        max_batch_index = int(step_tensor.item())
        print(f"Node: {NODE_RANK} Rank: {rank} num_batches {num_batches} max_batch_index {max_batch_index}")

        for batch_index, batch_data in enumerate(data_loader):
            loss_to_log.setdefault("train/epoch", []).append(epoch+1)
            loss_to_log.setdefault("train/batch_index", []).append(batch_index+1)

            t_temp_1_lr = time.time()
            lr_decay_iters = int(num_batches * args.epoch)
            warmup_iters = int(num_batches * args.epoch * args.warmup_ratio) if args.warmup_ratio else args.warmup_iters
            lr = get_lr(iter_num,
                        args.min_lr,
                        args.warmup_iters,
                        args.learning_rate,
                        lr_decay_iters) if args.decay_lr else args.learning_rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            t_temp_1_lr = time.time() - t_temp_1_lr
            t_temp_1_sum_lr += t_temp_1_lr

            t_temp_3_batchdata = time.time()
            input_gene_ids = batch_data["genes"].to(device)
            input_gene_values = batch_data["values"].to(device)
            input_gene_embeddings = batch_data["esm_embeddings"].to(device)
            input_gene_desc_embeddings = batch_data["desc_embeddings"].to(device)
            input_gene_dna_embeddings = batch_data["dna_embeddings"].to(device)
            target_values = batch_data["target_values"].to(device)
            # print(f"rank: {rank} batch done")
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
            # print(f"rank: {rank} batch to device done")
            t_temp_3_batchdata = time.time() - t_temp_3_batchdata
            t_temp_3_sum_batchdata += t_temp_3_batchdata
            # t_temp_4_sync = time.time()
            # if ddp:
            #     model.require_backward_grad_sync = ((batch_index + 1) % args.gradient_accumulation_steps == 0)
            #     print(f"rank: {rank} grad_sync done")
            # t_temp_4_sync = time.time() - t_temp_4_sync
            # t_temp_4_sum_sync += t_temp_4_sync
            t_temp_5_forward = time.time()
            with ctx:
                outputs = model(
                    src=input_gene_ids,
                    values=input_gene_values,
                    esm_embeddings=input_gene_embeddings,
                    desc_embeddings=input_gene_desc_embeddings,
                    dna_embeddings=input_gene_dna_embeddings,
                    batch_labels=input_batch_labels if config.use_batch_labels else None,
                    species_labels=input_species_labels if config.use_species_labels else None,
                    tissue_labels=input_tissue_labels if config.use_tissue_labels else None,
                    seqmethod_labels=input_seqmethod_labels if config.use_seqmethod_labels else None,
                    disease_labels=input_disease_labels if config.use_disease_labels else None,
                    sex_labels=input_sex_labels if config.use_sex_labels else None,
                    age_labels=input_age_labels if config.use_age_labels else None,
                    CLS=False,
                    MVC=True,
                    output_hidden_states=True,
                    output_attentions=False, # can not both turned on with _attn_implementation of sdpa
                    )
                # print(f"rank: {rank} outputs done")
                mask_positions = input_gene_values.eq(-1)
                t_temp_5_forward = time.time() - t_temp_5_forward
                t_temp_5_sum_forward += t_temp_5_forward

                t_temp_6_loss_gep = time.time()
                # compute loss
                loss = 0.0
                loss_gep = masked_mse_loss(outputs["model_output"],
                                           target_values,
                                           mask_positions)
                loss += loss_gep
                loss_to_log.setdefault("train/GEP", []).append(loss_gep.item())
                # print(f"rank: {rank} loss_gep done")
                t_temp_6_loss_gep = time.time() - t_temp_6_loss_gep
                t_temp_6_sum_loss_gep += t_temp_6_loss_gep

                t_temp_7_loss_gep_zero_prob = time.time()
                if config.explicit_zero_prob:
                    if torch.isnan(outputs["model_zero_prob"]).any():
                        raise ValueError("There are nan values in zero prob output!")
                    loss_zero_prob = criterion_neg_log_bernoulli(outputs["model_zero_prob"],
                                                                 target_values,
                                                                 mask_positions)
                    loss += loss_zero_prob
                    loss_to_log.setdefault("train/nzlp", []).append(loss_gep.item())

                    # print(f"rank: {rank} loss_zero_prob done")
                if "mvc_output" in outputs:
                    loss_gepc = masked_mse_loss(outputs["mvc_output"],
                                                target_values,
                                                mask_positions)
                    loss += loss_gepc
                    loss_to_log.setdefault("train/GEPC", []).append(loss_gep.item())

                    # print(f"rank: {rank} mvc_output done")
                if "mvc_output" in outputs and config.explicit_zero_prob:
                    loss_gepc_zero_prob = criterion_neg_log_bernoulli(outputs["mvc_zero_probs"],
                                                                      target_values,
                                                                      mask_positions)
                    loss += loss_gepc_zero_prob
                    loss_to_log.setdefault("train/GEPC_nzlp", []).append(loss_gep.item())

                    # print(f"rank: {rank} loss_gepc_zero_prob done")
                loss = loss / args.gradient_accumulation_steps
                t_temp_7_loss_gep_zero_prob = time.time() - t_temp_7_loss_gep_zero_prob
                t_temp_7_sum_loss_gep_zero_prob += t_temp_7_loss_gep_zero_prob

            # backward and optimization
            t_temp_8_backward = time.time()
            scaler.scale(loss).backward()
            # print(f"rank: {rank} scaler backward done")
            t_temp_8_backward = time.time() - t_temp_8_backward
            t_temp_8_sum_backward += t_temp_8_backward

            # for name, param in model.named_parameters():
            #     if param.grad is None:
            #         print(f"WARNING: {name} grad is None! ^^^^^^^^^^^^^^^^^^^^^^")
            t_temp_9_step_update_grad = time.time()
            t_temp_10_sync = 0
            if (batch_index + 1) % args.gradient_accumulation_steps == 0 or (batch_index + 1) == num_batches or (batch_index + 1) >= max_batch_index:
                if grad_clip != 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                # # to increase the sync ratio
                t_temp_10_sync = time.time()
                all_reduce_count = 0
                for param in model.parameters():
                    if param.grad is not None:
                        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                        param.grad.data /= float(world_size)
                        all_reduce_count += 1
                t_temp_10_sync = time.time() - t_temp_10_sync
                t_temp_10_sum_sync += t_temp_10_sync

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                # print(f"rank: {rank} step update zero_grad done")

                if (batch_index + 1) % save_step_interval == 0:
                    lossf = loss.item() * args.gradient_accumulation_steps
                    if ddp:
                        save_model(model.module, optimizer, epoch, batch_index + 1, lossf, out_dir, rank, NODE_RANK, s3_remote_dir_path=args.s3_remote_dir_path)
                    else:
                        save_model(model, optimizer, epoch, batch_index + 1, lossf, out_dir, rank, NODE_RANK, s3_remote_dir_path=args.s3_remote_dir_path)
                    save_log_to_s3(args, out_dir, NODE_RANK, rank)
            else:
                pass
                # print(f"rank: {rank} not in step update zero_grad")
            t_temp_9_step_update_grad = time.time() - t_temp_9_step_update_grad
            t_temp_9_sum_step_update_grad += t_temp_9_step_update_grad


            t_temp_1 = time.time()
            dt = t_temp_1 - t_temp_0
            time_checker = (f"1_lr {t_temp_1_lr:.4f} "
                            f"2_epoch_dataloder {t_temp_2_data:.4f} "
                            f"3_step_batched_data {t_temp_3_batchdata:.4f} "
                            # f"4_sync {t_temp_4_sync:.4f} "
                            f"5_forward {t_temp_5_forward:.4f} "
                            f"6_loss_gep {t_temp_6_loss_gep:.4f} "
                            f"7_loss_gep_zero_prob {t_temp_7_loss_gep_zero_prob:.4f} "
                            f"8_backward {t_temp_8_backward:.4f} "
                            f"9_step_update_grad {t_temp_9_step_update_grad:.4f} "
                            f"10_sync {t_temp_10_sync:.4f}")
            t_temp_0 = t_temp_1

            lossf = loss.item() * args.gradient_accumulation_steps
            if iter_num % 100 == 0 and iter_num != 0:
                mfu = raw_model.estimate_mfu(args.batch_size * args.gradient_accumulation_steps, dt)
                runing_mfu = mfu if runing_mfu == -1.0 else 0.9 * runing_mfu + 0.1 * mfu
            log_info = f"Node: {NODE_RANK}, rank: {rank}, [e: {epoch+1}, {batch_index + 1}/{num_batches}], loss: {lossf:.4f}, time {dt * 1000:.2f}ms, mfu: {runing_mfu * 100:.2f}%, {time_checker}"
            print(log_info)
            write_to_log(log_info, local_log_file)

            total_loss += loss.item()
            # print(f"rank: {rank} total_loss")
            loss_to_log.setdefault("train/total_loss", []).append(loss_gep.item())
            iter_num += 1

            if (batch_index + 1) >= max_batch_index:
                break

        total_loss = total_loss * args.gradient_accumulation_steps / num_batches
        loss_info = f"Node: {NODE_RANK}, Rank: {rank}, Epoch [{epoch + 1}/{args.epoch}], average loss is: {total_loss:.4f} | Learning rate is: {lr}"
        print(loss_info)
        write_to_log(loss_info, local_log_file)
        time_checker = (
                        f"Node: {NODE_RANK} "
                        f"Rank: {rank} "
                        f"1_lr {t_temp_1_sum_lr:.4f} "
                        f"2_epoch_dataloder {t_temp_2_sum_data:.4f} "
                        f"3_step_batched_data {t_temp_3_sum_batchdata:.4f} "
                        # f"4_sync {t_temp_4_sum_sync:.4f} "
                        f"5_forward {t_temp_5_sum_forward:.4f} "
                        f"6_loss_gep {t_temp_6_sum_loss_gep:.4f} "
                        f"7_loss_gep_zero_prob {t_temp_7_sum_loss_gep_zero_prob:.4f} "
                        f"8_backward {t_temp_8_sum_backward:.4f} "
                        f"9_step_update_grad {t_temp_9_sum_step_update_grad:.4f} "
                        f"10_sync {t_temp_10_sum_sync:.4f}")
        print(time_checker)
        write_to_log(time_checker, local_log_file)
        total_loss = 0

    pd.DataFrame(loss_to_log).to_csv(os.path.join(out_dir, f"loss_to_log.{NODE_RANK}-{rank}.txt"), index=False)
    save_log_to_s3(args, out_dir, NODE_RANK, rank)

    NODE_RANK = os.getenv('NODE_RANK') if os.getenv('NODE_RANK') else os.getenv('node_rank')
    if ddp:
        save_model(model.module, optimizer, args.epoch+1, 0, 0, out_dir, rank, NODE_RANK, s3_remote_dir_path=args.s3_remote_dir_path)
    else:
        save_model(model, optimizer, args.epoch+1, 0, 0, out_dir, rank, NODE_RANK, s3_remote_dir_path=args.s3_remote_dir_path)
    save_log_to_s3(args, out_dir, NODE_RANK, rank)

def save_log_to_s3(args, out_dir, NODE_RANK, rank):
    try:
        mox.file.mk_dir(args.s3_remote_dir_path)
        mox.file.copy(os.path.join(out_dir,
                                   f"loss_to_log.{NODE_RANK}-{rank}.txt"),
                      args.s3_remote_dir_path.strip("/") + "/" + f"loss_to_log.{NODE_RANK}-{rank}.txt")
        mox.file.copy(os.path.join(out_dir,
                                   f"log.{NODE_RANK}-{rank}.txt"),
                      args.s3_remote_dir_path.strip("/") + "/" + f"log.{NODE_RANK}-{rank}.txt")
        print(f'torch.save and to S3: {args.s3_remote_dir_path.strip("/") + "/" + f"loss_to_log.{NODE_RANK}-{rank}.txt"} {args.s3_remote_dir_path.strip("/") + "/" + f"log.{NODE_RANK}-{rank}.txt"}')
    except Exception as e:
        print(e)

def save_model(model, optimizer, epoch, step, loss, out_dir, local_rank=None, NODE_RANK=None, savepath=None, s3_remote_dir_path=None):
    if NODE_RANK is None:
        NODE_RANK = os.getenv('NODE_RANK') if os.getenv('NODE_RANK') else os.getenv('node_rank')
    if not savepath:
        savepath = "SC-node-{node:02d}-rank-{rank:02d}-epoch-{epoch:02d}-step-{step}-loss-{loss:.6f}.pt"
    save_path = savepath.format(
        node=int(NODE_RANK),
        rank=int(local_rank),
        epoch=int(epoch),
        step=str(step),
        loss=loss)
    save_other_path = save_path.replace(".pt", ".optimizer.pt")
    with torch.no_grad():
        torch.save(model.state_dict(), os.path.join(out_dir, save_path))
        torch.save(optimizer.state_dict(), os.path.join(out_dir, save_other_path))
    if s3_remote_dir_path and (s3_remote_dir_path.startswith("s3://") or s3_remote_dir_path.startswith("obs://")):
        mox.file.mk_dir(s3_remote_dir_path)
        mox.file.copy(os.path.join(out_dir, save_path), s3_remote_dir_path.strip("/") + "/" + save_path)
        mox.file.copy(os.path.join(out_dir, save_other_path), s3_remote_dir_path.strip("/") + "/" + save_other_path)
        print(f'torch.save and to S3: {s3_remote_dir_path.strip("/") + "/" + save_path} {s3_remote_dir_path.strip("/") + "/" + save_other_path}')
        os.remove(os.path.join(out_dir, save_path))
        os.remove(os.path.join(out_dir, save_other_path))
    else:
        print(f"torch.save: {save_path} {save_other_path}")

def write_to_log(c, path):
    with open(path, "a") as f:
        f.write(c+"\n")


def main(args):
    data_path = args.data_path
    emb_path = args.emb_path
    out_dir = args.out_path
    seq_len = args.seq_len
    batch_size = args.batch_size  # if gradient_accumulation_steps > 1, this is the mini-batch size
    num_epochs = args.epoch

    # Learning rate setting
    learning_rate = args.learning_rate
    # lr_decay_iters = args.lr_decay_iters
    # max_iters = args.max_iters
    min_lr = args.min_lr
    decay_lr = args.decay_lr
    warmup_iters = args.warmup_iters
    warmup_ratio = args.warmup_ratio
    weight_decay = args.weight_decay
    beta1 = args.beta1
    beta2 = args.beta2

    grad_clip = args.grad_clip  # clip gradients at this value, or disable if == 0.0
    compile = args.compile  # complie model if pytorch > 2.0
    gradient_accumulation_steps = args.gradient_accumulation_steps  # used to simulate larger batch sizes
    backend = args.backend
    device = args.device
    device_type = args.device_type

    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[
        dtype]  # note: float16 data type will automatically use a GradScaler
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # time of start
    t0 = time.time()
    src = np.arange(1, seq_len + 1)
    #######################################
    # original reading section
    esm_embeddings = np.load(emb_path + "/2nd_run_macrogene_features_sum_esm2.npy")
    desc_embeddings = np.load(emb_path + "/2nd_run_macrogene_features_sum_gene_desc.npy")
    dna_embeddings = np.load(emb_path + "/2nd_run_macrogene_features_sum_dnaseq.npy")
    #######################################

    # Build vocabulary
    gene_ids = ["gene_" + str(id) for id in range(1, seq_len + 1)]
    cls_token = ["<cls>"]
    special_tokens = ["<cls>", "<sep>", "<pad>", "<mask>"]
    combined_vocab = cls_token + gene_ids + special_tokens
    vocab = {token: idx for idx, token in enumerate(combined_vocab)}
    config = build_bertconfig(vocab, seq_len, args)
    collate_fn = CustomCollate_3GeneEmb(config=config,
                                        genes=src,
                                        esm_embeddings=esm_embeddings,
                                        desc_embeddings=desc_embeddings,
                                        dna_embeddings=dna_embeddings)
    
    print("\n=== build model ===")
    print(config)

    out_dir = out_dir.format(hidden_size=config.hidden_size, num_hidden_layers=config.num_hidden_layers,
                             num_attention_heads=config.num_attention_heads, hidden_dropout_prob=config.hidden_dropout_prob,
                             learning_rate=learning_rate, min_lr=min_lr, weight_decay=weight_decay, warmup_ratio=warmup_ratio)
    args.s3_remote_dir_path = os.path.join("/".join(args.s3_remote_dir_path.split("/")[:-1]), out_dir + "_" + args.s3_remote_dir_path.split("/")[-1])
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.exists(out_dir):
        raise FileNotFoundError(f"Output directory {out_dir} does not exist!")
        sys.exit(1)
    print(os.path.abspath(out_dir))
    print(args.s3_remote_dir_path)

    # set up for multiple GPUs run
    ddp, rank, local_rank, master_process, world_size = setup_ddp(backend='hccl', device_type='npu')
    device = f"npu:{local_rank}"
    tokens_per_epoch = gradient_accumulation_steps * world_size * batch_size * seq_len
    print(f"ddp: {ddp}, rank: {rank}, local_rank: {local_rank}, world_size: {world_size}, master_process: {master_process}, device: {device}, tokens_per_epoch: {tokens_per_epoch}")

    # Since we do not use DistributedSampler to distribute data, so commenting it
    train_data_filelist = get_files(data_path, args.num_of_used_data)  # get all parquet files, use for PreindexedParquetDataset
    # train_data_filelist = load_data_for_total(data_path) # get all parquet files, use for ParquetDataset
    train_sampler = DistributedFileSampler(train_data_filelist,
                                     num_replicas=world_size,
                                     rank=rank,
                                     drop_last=True)

    model = BERTForPreTraining(config).to(device)
    # optimizer and initialize a GradSclaer.
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2))
    # NanoGPT's way to initialize optimizer. But using this with turn use_fused = False in model
    # optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    scaler = GradScaler(enabled=(dtype == 'float16'))
    # compile the model
    if compile:
        if master_process:
            print(f"{compile} compiling the model...(take a ~minute)")
        unoptimized_model = model
        model = torch.compile(model)  # requires P
    if ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    ctx = nullcontext() if device_type == 'cpu' else torch.autocast(device_type=device_type, dtype=ptdtype)
    train_loop(args, model, ddp, rank, optimizer, train_data_filelist, train_sampler, device, config, ctx, scaler, grad_clip, out_dir, collate_fn, world_size)

    t1 = time.time()
    print(f"Complete pretraining! Running time is {t1 - t0:.2f}s.")


def argumentparser():
    parser = argparse.ArgumentParser(description="Distributed Multi-Node Multi-GPU parquet training")
    parser.add_argument("--data_path",
                        type=str,
                        required=True)
    parser.add_argument("--num_of_used_data",
                        type=int,
                        required=0)
    parser.add_argument("--emb_path",
                        type=str,
                        required=True)
    parser.add_argument("--seq_len",
                        type=int,
                        required=True)
    parser.add_argument("--out_path",
                        type=str,
                        default="hs_{hidden_size}_nh_{num_hidden_layers}_na_{num_attention_heads}_hdp_{hidden_dropout_prob}_lr_{learning_rate}_mlr_{min_lr}_wd_{weight_decay}_wr_{warmup_ratio}")
    parser.add_argument("--batch_size",
                        type=int,
                        default=64)
    parser.add_argument("--epoch",
                        type=int,
                        default=10)
    parser.add_argument("--gradient_accumulation_steps",
                        type=int,
                        default=4)
    parser.add_argument("--learning_rate",
                        type=float,
                        default=1e-4)
    # parser.add_argument('--lr_decay_iters',
    #                     type=int,
    #                     default=600000)
    # parser.add_argument('--max_iters',
    #                     type=int,
    #                     default=999999999)
    parser.add_argument('--min_lr',
                        type=float,
                        default=6e-5)
    parser.add_argument('--decay_lr',
                        type=bool,
                        default=True)
    parser.add_argument('--warmup_iters',
                        type=int,
                        default=2000)
    parser.add_argument('--warmup_ratio',
                        type=float,
                        default=0.01,
                        help='if warmup_ratio, then discard warmup_iters')
    parser.add_argument('--weight_decay',
                        type=float,
                        default=1e-1)
    parser.add_argument('--save_data_interval',
                        type=int,
                        default=5120000)
    parser.add_argument('--beta1',
                        type=float,
                        default=0.9)
    parser.add_argument('--beta2',
                        type=float,
                        default=0.95)
    parser.add_argument('--grad_clip',
                        type=float,
                        default=1.0)
    parser.add_argument('--compile',
                        type=bool,
                        default=False)
    parser.add_argument('--backend',
                        type=str,
                        default='hccl')
    parser.add_argument('--device',
                        type=str,
                        default='npu')
    parser.add_argument('--device_type',
                        type=str,
                        default='npu')
    parser.add_argument('--s3_remote_dir_path',
                        type=str,
                        default="")
    parser.add_argument('--hidden_size',
        type=int,
        default=1280)
    parser.add_argument('--num_hidden_layers',
        type=int,
        default=24)
    parser.add_argument('--num_attention_heads',
        type=int,
        default=20)
    parser.add_argument('--intermediate_size',
        type=int,
        default=5120)
    parser.add_argument('--hidden_act',
        type=str,
        default="gelu")
    parser.add_argument('--hidden_dropout_prob',
        type=float,
        default=0.1)
    parser.add_argument('--cell_hidden_size',
        type=int,
        default=128)
    parser.add_argument('--attention_probs_dropout_prob',
        type=float,
        default=0.1)
    parser.add_argument('--type_vocab_size',
        type=int,
        default=2)
    parser.add_argument('--initializer_range',
        type=float,
        default=0.02)
    parser.add_argument('--layer_norm_eps',
        type=float,
        default=1e-12)
    parser.add_argument('--_attn_implementation',
        type=str,
        default="sdpa")
    parser.add_argument('--use_batch_labels',
        type=str,
        default="False")
    parser.add_argument('--num_batch_labels',
        type=int,
        default=12028)
    parser.add_argument('--use_species_labels',
        type=str,
        default="True")
    parser.add_argument('--num_species_labels',
        type=int,
        default=11)
    parser.add_argument('--use_tissue_labels',
        type=str,
        default="True")
    parser.add_argument('--num_tissue_labels',
        type=int,
        default=154)
    parser.add_argument('--use_seqmethod_labels',
        type=str,
        default="True")
    parser.add_argument('--num_seqmethod_labels',
        type=int,
        default=28)
    parser.add_argument('--use_disease_labels',
        type=str,
        default="True")
    parser.add_argument('--num_disease_labels',
        type=int,
        default=143)
    parser.add_argument('--use_age_labels',
        type=str,
        default="True")
    parser.add_argument('--num_age_labels',
        type=int,
        default=5)
    parser.add_argument('--use_sex_labels',
        type=str,
        default="True")
    parser.add_argument('--num_sex_labels',
        type=int,
        default=3)
    parser.add_argument('--cell_emb_style',
        type=str,
        default="cls")
    parser.add_argument('--chunk_size_feed_forward',
        type=int,
        default=0)
    parser.add_argument('--explicit_zero_prob',
        type=str,
        default="True")


    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argumentparser()
    # 使用json格式化
    print("\n=== sys.argv ===")
    print(sys.argv)
    time.sleep(10)
    print("\n=== JSON格式 ===")
    print(json.dumps(vars(args), indent=4))
    time.sleep(10)
    main(args)

"""
# para 1.1 - 1.4
--learning_rate=0.0005 --min_lr=0.00006 --weight_decay=0.1 --warmup_ratio=0.05 --hidden_dropout_prob=0.1
--learning_rate=0.0001 --min_lr=0.00006 --weight_decay=0.1 --warmup_ratio=0.05 --hidden_dropout_prob=0.1
--learning_rate=0.00005 --min_lr=0.00006 --weight_decay=0.1 --warmup_ratio=0.05 --hidden_dropout_prob=0.1
--learning_rate=0.00001 --min_lr=0.00006 --weight_decay=0.1 --warmup_ratio=0.05 --hidden_dropout_prob=0.1

# para 2.1 - 2.3
--learning_rate=0.0001 --min_lr=0.000001 --weight_decay=0.1 --warmup_ratio=0.05 --hidden_dropout_prob=0.1
--learning_rate=0.0001 --min_lr=0.00006 --weight_decay=0.1 --warmup_ratio=0.05 --hidden_dropout_prob=0.1
--learning_rate=0.0001 --min_lr=0.00001 --weight_decay=0.1 --warmup_ratio=0.05 --hidden_dropout_prob=0.1

# para 3.1 - 3.3
--learning_rate=0.00001 --min_lr=0.00006 --weight_decay=0.1 --warmup_ratio=0.01 --hidden_dropout_prob=0.1
--learning_rate=0.00001 --min_lr=0.00006 --weight_decay=0.1 --warmup_ratio=0.02 --hidden_dropout_prob=0.1
--learning_rate=0.00001 --min_lr=0.00006 --weight_decay=0.1 --warmup_ratio=0.05 --hidden_dropout_prob=0.1

# para 4.1 - 4.4
--learning_rate=0.00001 --min_lr=0.00006 --weight_decay=0.1 --warmup_ratio=0.01 --hidden_dropout_prob=0
--learning_rate=0.00001 --min_lr=0.00006 --weight_decay=0.1 --warmup_ratio=0.01 --hidden_dropout_prob=0.01
--learning_rate=0.00001 --min_lr=0.00006 --weight_decay=0.1 --warmup_ratio=0.01 --hidden_dropout_prob=0.05
--learning_rate=0.00001 --min_lr=0.00006 --weight_decay=0.1 --warmup_ratio=0.01 --hidden_dropout_prob=0.1



# para 1.1 - 1.4 SpeciesLLM-exp2-para1_4-fixminlr
--learning_rate=0.0005  --min_lr=0.00006  --weight_decay=0.1 --warmup_ratio=0.05 --hidden_dropout_prob=0.1 # 不用重跑
--learning_rate=0.0001  --min_lr=0.00006  --weight_decay=0.1 --warmup_ratio=0.05 --hidden_dropout_prob=0.1 # 不用重跑
--learning_rate=0.00005 --min_lr=0.000005 --weight_decay=0.1 --warmup_ratio=0.05 --hidden_dropout_prob=0.1
--learning_rate=0.00001 --min_lr=0.000001 --weight_decay=0.1 --warmup_ratio=0.05 --hidden_dropout_prob=0.1

# para 2.1 - 2.3
--learning_rate=0.0001  --min_lr=0.000001 --weight_decay=0.1 --warmup_ratio=0.05 --hidden_dropout_prob=0.1 # 不用重跑
--learning_rate=0.0001  --min_lr=0.00006  --weight_decay=0.1 --warmup_ratio=0.05 --hidden_dropout_prob=0.1 # 不用重跑
--learning_rate=0.0001  --min_lr=0.00001  --weight_decay=0.1 --warmup_ratio=0.05 --hidden_dropout_prob=0.1 # 不用重跑

# para 3.1 - 3.3
--learning_rate=0.00001 --min_lr=0.000001 --weight_decay=0.1 --warmup_ratio=0.01 --hidden_dropout_prob=0.1
--learning_rate=0.00001 --min_lr=0.000001 --weight_decay=0.1 --warmup_ratio=0.02 --hidden_dropout_prob=0.1
--learning_rate=0.00001 --min_lr=0.000001 --weight_decay=0.1 --warmup_ratio=0.05 --hidden_dropout_prob=0.1

# para 4.1 - 4.4
--learning_rate=0.00001 --min_lr=0.000001 --weight_decay=0.1 --warmup_ratio=0.01 --hidden_dropout_prob=0
--learning_rate=0.00001 --min_lr=0.000001 --weight_decay=0.1 --warmup_ratio=0.01 --hidden_dropout_prob=0.01
--learning_rate=0.00001 --min_lr=0.000001 --weight_decay=0.1 --warmup_ratio=0.01 --hidden_dropout_prob=0.05
--learning_rate=0.00001 --min_lr=0.000001 --weight_decay=0.1 --warmup_ratio=0.01 --hidden_dropout_prob=0.1
"""