#!/usr/bin/env python3
"""Probe whether DDP backward synchronizes gradients on the active backend.

Run on Ascend NPU with torchrun, for example:

    torchrun --standalone --nproc_per_node=2 \
      scripts/ddp_grad_sync_probe.py --device npu --backend hccl

Optional CPU sanity check on a development machine:

    torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29591 \
      scripts/ddp_grad_sync_probe.py --device cpu --backend gloo

The probe uses a one-parameter model with rank-specific targets. Without
gradient synchronization, rank r gets gradient -2 * (r + 1). With correct DDP
gradient averaging, every rank gets -(world_size + 1).
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import sys
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP


@dataclass(frozen=True)
class Case:
    name: str
    use_ddp: bool
    expect: str
    manual_all_reduce: bool = False
    require_backward_grad_sync: bool | None = None
    use_no_sync_context: bool = False


CASES = {
    "plain_no_ddp": Case(
        name="plain_no_ddp",
        use_ddp=False,
        expect="local",
    ),
    "manual_no_ddp": Case(
        name="manual_no_ddp",
        use_ddp=False,
        manual_all_reduce=True,
        expect="avg",
    ),
    "ddp_backward": Case(
        name="ddp_backward",
        use_ddp=True,
        expect="avg",
    ),
    "ddp_require_flag_false": Case(
        name="ddp_require_flag_false",
        use_ddp=True,
        require_backward_grad_sync=False,
        expect="local",
    ),
    "ddp_no_sync_context": Case(
        name="ddp_no_sync_context",
        use_ddp=True,
        use_no_sync_context=True,
        expect="local",
    ),
}

DEFAULT_CASES = [
    "plain_no_ddp",
    "manual_no_ddp",
    "ddp_backward",
    "ddp_require_flag_false",
    "ddp_no_sync_context",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate whether DDP loss.backward() really synchronizes parameter "
            "gradients across ranks."
        )
    )
    parser.add_argument(
        "--device",
        choices=("npu", "cpu"),
        default="npu",
        help="Device type to test. Use npu/hccl on Ascend, cpu/gloo for local sanity checks.",
    )
    parser.add_argument(
        "--backend",
        default="hccl",
        help="torch.distributed backend. Use hccl for Ascend NPU, gloo for CPU.",
    )
    parser.add_argument(
        "--init-method",
        default=None,
        help=(
            "Optional explicit init_method, for example tcp://127.0.0.1:29591. "
            "When omitted, torchrun's environment rendezvous is used."
        ),
    )
    parser.add_argument(
        "--cases",
        default="all",
        help=(
            "Comma-separated case list, or 'all'. Available: "
            + ", ".join(DEFAULT_CASES)
        ),
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-5,
        help="Absolute tolerance for gradient comparisons.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-5,
        help="Relative tolerance for gradient comparisons.",
    )
    parser.add_argument(
        "--gradient-as-bucket-view",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass gradient_as_bucket_view to DDP. Default matches current training entry.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each rank's local result before rank-0 summary.",
    )
    return parser.parse_args()


def selected_cases(raw_cases: str) -> list[Case]:
    if raw_cases == "all":
        names = DEFAULT_CASES
    else:
        names = [item.strip() for item in raw_cases.split(",") if item.strip()]
    unknown = [name for name in names if name not in CASES]
    if unknown:
        raise SystemExit(
            f"unknown case(s): {', '.join(unknown)}; available: {', '.join(DEFAULT_CASES)}"
        )
    return [CASES[name] for name in names]


def env_int(name: str) -> int:
    value = os.environ.get(name)
    if value is None:
        raise SystemExit(f"{name} is not set; run this script with torchrun")
    try:
        return int(value)
    except ValueError as exc:
        raise SystemExit(f"{name} must be an integer, got {value!r}") from exc


def setup_device(args: argparse.Namespace, local_rank: int) -> torch.device:
    if args.device == "npu":
        try:
            import torch_npu  # noqa: F401
        except ModuleNotFoundError as exc:
            raise SystemExit(
                "torch_npu is not importable; run on the Ascend training environment "
                "or use --device cpu --backend gloo for a local sanity check"
            ) from exc
        torch.npu.set_device(local_rank)
        return torch.device(f"npu:{local_rank}")
    return torch.device("cpu")


def synchronize_device(device: torch.device) -> None:
    if device.type == "npu":
        torch.npu.synchronize()


def make_model(
    args: argparse.Namespace,
    device: torch.device,
    local_rank: int,
    use_ddp: bool,
) -> nn.Module:
    model = nn.Linear(1, 1, bias=False).to(device)
    with torch.no_grad():
        model.weight.fill_(0.0)
    if not use_ddp:
        return model

    device_ids = [local_rank] if device.type == "npu" else None
    return DDP(
        model,
        device_ids=device_ids,
        broadcast_buffers=False,
        gradient_as_bucket_view=args.gradient_as_bucket_view,
    )


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def expected_local_grad(rank: int) -> float:
    return -2.0 * float(rank + 1)


def expected_avg_grad(world_size: int) -> float:
    return -float(world_size + 1)


def expected_sum_grad(world_size: int) -> float:
    return -float(world_size * (world_size + 1))


def close(a: float, b: float, *, atol: float, rtol: float) -> bool:
    return math.isclose(a, b, abs_tol=atol, rel_tol=rtol)


def all_close(values: Iterable[float], expected: Iterable[float], *, atol: float, rtol: float) -> bool:
    return all(close(value, want, atol=atol, rtol=rtol) for value, want in zip(values, expected))


def gather_scalar(value: torch.Tensor, world_size: int) -> list[float]:
    packed = value.detach().reshape(1).to(dtype=torch.float32)
    gathered = [torch.zeros_like(packed) for _ in range(world_size)]
    dist.all_gather(gathered, packed)
    return [float(item.cpu().item()) for item in gathered]


def run_case(
    args: argparse.Namespace,
    case: Case,
    rank: int,
    local_rank: int,
    world_size: int,
    device: torch.device,
) -> dict[str, object]:
    model = make_model(args, device, local_rank, use_ddp=case.use_ddp)
    if case.require_backward_grad_sync is not None:
        model.require_backward_grad_sync = case.require_backward_grad_sync

    x = torch.ones((1, 1), dtype=torch.float32, device=device)
    target = torch.tensor([[float(rank + 1)]], dtype=torch.float32, device=device)

    sync_context = (
        model.no_sync()
        if case.use_ddp and case.use_no_sync_context
        else contextlib.nullcontext()
    )
    with sync_context:
        output = model(x)
        loss = (output - target).pow(2).sum()
        loss.backward()

    raw_model = unwrap_model(model)
    grad = raw_model.weight.grad
    if grad is None:
        raise RuntimeError(f"{case.name}: gradient is None on rank {rank}")

    if case.manual_all_reduce:
        dist.all_reduce(grad, op=dist.ReduceOp.SUM)
        grad.div_(float(world_size))

    synchronize_device(device)
    observed = gather_scalar(grad, world_size)

    expected_local = [expected_local_grad(item) for item in range(world_size)]
    expected_avg = expected_avg_grad(world_size)
    expected_sum = expected_sum_grad(world_size)

    if case.expect == "local":
        passed = all_close(observed, expected_local, atol=args.atol, rtol=args.rtol)
        expected_label = f"local per-rank {expected_local}"
    elif case.expect == "avg":
        passed = all_close(
            observed,
            [expected_avg] * world_size,
            atol=args.atol,
            rtol=args.rtol,
        )
        expected_label = f"global average {expected_avg}"
    else:
        raise RuntimeError(f"unsupported expectation: {case.expect}")

    same_across_ranks = all(close(value, observed[0], atol=args.atol, rtol=args.rtol) for value in observed)
    looks_like_sum = same_across_ranks and close(observed[0], expected_sum, atol=args.atol, rtol=args.rtol)

    if args.verbose:
        print(
            f"[rank {rank}] case={case.name} grad={float(grad.cpu().item()):.8g} "
            f"target={float(target.cpu().item()):.8g}",
            flush=True,
        )

    dist.barrier()
    return {
        "case": case.name,
        "observed": observed,
        "expected": expected_label,
        "passed": passed,
        "same_across_ranks": same_across_ranks,
        "looks_like_sum": looks_like_sum,
    }


def print_rank0_result(
    result: dict[str, object],
    rank: int,
    world_size: int,
) -> None:
    if rank != 0:
        return
    status = "PASS" if result["passed"] else "FAIL"
    note = ""
    if result["looks_like_sum"]:
        note = " NOTE: synchronized value looks like a SUM, not an average"
    print(
        f"{status} {result['case']}: observed={result['observed']} "
        f"expected={result['expected']}{note}",
        flush=True,
    )


def main() -> int:
    args = parse_args()
    cases = selected_cases(args.cases)

    rank = env_int("RANK")
    local_rank = env_int("LOCAL_RANK")
    world_size = env_int("WORLD_SIZE")
    if world_size < 2:
        raise SystemExit("world_size must be at least 2 to validate cross-rank gradient sync")

    device = setup_device(args, local_rank)
    if args.init_method:
        dist.init_process_group(
            backend=args.backend,
            init_method=args.init_method,
            rank=rank,
            world_size=world_size,
        )
    else:
        dist.init_process_group(backend=args.backend)

    try:
        if rank == 0:
            print(
                f"backend={args.backend} device={device.type} "
                f"gradient_as_bucket_view={args.gradient_as_bucket_view}",
                flush=True,
            )
        if rank == 0:
            print(f"world_size={world_size}", flush=True)
        results = []
        for case in cases:
            result = run_case(args, case, rank, local_rank, world_size, device)
            results.append(result)
            print_rank0_result(result, rank, world_size)
        failed = [result["case"] for result in results if not result["passed"]]
        if failed and rank == 0:
            print(f"failed_cases={failed}", flush=True)
        return 1 if failed else 0
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())
