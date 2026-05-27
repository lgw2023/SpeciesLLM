#!/usr/bin/env python3
"""
NPU AICore 填充脚本 —— 在训练任务的数据加载间隙填补 AICore 空闲。

设计目标:
  - 极低显存占用 (默认 ~24MB/卡，float16 下 3 个 1024x1024 矩阵)
  - 不干扰已有训练: 预分配固定 tensor，零额外显存申请
  - 自适应退让: 定期检测剩余显存，低于阈值自动暂停
  - 可控强度: 通过 --burst / --sleep_ms 调节计算密度

用法:
  # 单卡 (在 NPU:3 上跑)
  python npu_aicore_filler.py --device 3

  # 多卡 (每卡一个进程)
  for i in 0 1 2 3 4 5 6 7; do
    python npu_aicore_filler.py --device $i &
  done

  # 轻量模式 (更少计算, 更多 sleep)
  python npu_aicore_filler.py --device 0 --burst 50 --sleep_ms 20

  # 停止: Ctrl+C 或 kill, 会自动清理显存
"""

import os
import sys
import time
import signal
import argparse

import torch

try:
    import torch_npu
    HAS_NPU = True
except ImportError:
    HAS_NPU = False


def get_free_memory_gb(device):
    if HAS_NPU:
        props = torch_npu.npu.get_device_properties(device)
        total = props.total_memory
        reserved = torch_npu.npu.memory_reserved(device)
        return (total - reserved) / (1024 ** 3)
    else:
        props = torch.cuda.get_device_properties(device)
        total = props.total_mem
        reserved = torch.cuda.memory_reserved(device)
        return (total - reserved) / (1024 ** 3)


def empty_cache():
    if HAS_NPU:
        torch_npu.npu.empty_cache()
    else:
        torch.cuda.empty_cache()


def synchronize(device):
    if HAS_NPU:
        torch_npu.npu.synchronize(device)
    else:
        torch.cuda.synchronize(device)


def parse_args():
    p = argparse.ArgumentParser(
        description="NPU AICore filler: 低显存占用的后台计算填充",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--device", type=int, default=0,
                   help="NPU/GPU device ID (默认 0)")
    p.add_argument("--matrix_size", type=int, default=1024,
                   help="矩阵维度 N (N×N matmul). 默认 1024 → ~24MB 显存")
    p.add_argument("--burst", type=int, default=200,
                   help="每轮连续 matmul 次数 (越大→AICore 越满, CPU 开销越低)")
    p.add_argument("--sleep_ms", type=float, default=5.0,
                   help="每轮之间的 sleep (ms). 0=全速; 越大→让出越多 AICore 给训练")
    p.add_argument("--min_free_gb", type=float, default=1.5,
                   help="剩余显存低于此值(GB)时暂停计算, 等待显存释放")
    p.add_argument("--check_interval", type=int, default=500,
                   help="每隔多少轮检查一次显存")
    p.add_argument("--dtype", type=str, default="float16",
                   choices=["float16", "bfloat16", "float32"],
                   help="计算精度. float16 最省显存且最能压 AICore")
    p.add_argument("--report_interval", type=int, default=2000,
                   help="每隔多少轮打印一次统计")
    p.add_argument("--max_seconds", type=int, default=0,
                   help="最大运行秒数, 0=无限制")
    return p.parse_args()


def main():
    args = parse_args()

    device_type = "npu" if HAS_NPU else "cuda"
    device = torch.device(f"{device_type}:{args.device}")
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]
    bytes_per_elem = 2 if dtype in (torch.float16, torch.bfloat16) else 4
    N = args.matrix_size
    mem_mb = 3 * N * N * bytes_per_elem / (1024 ** 2)

    # --- 安全检查 ---
    if HAS_NPU:
        torch_npu.npu.set_device(args.device)
    else:
        torch.cuda.set_device(args.device)

    free_gb = get_free_memory_gb(device)
    print(f"[AICore Filler] device={device}, dtype={args.dtype}, matrix={N}×{N}")
    print(f"[AICore Filler] 预计显存占用: {mem_mb:.1f} MB (3 个矩阵)")
    print(f"[AICore Filler] 当前剩余显存: {free_gb:.2f} GB")
    print(f"[AICore Filler] burst={args.burst}, sleep={args.sleep_ms}ms, min_free={args.min_free_gb}GB")

    if free_gb < args.min_free_gb:
        print(f"[AICore Filler] 剩余显存 {free_gb:.2f}GB < {args.min_free_gb}GB, 拒绝启动")
        sys.exit(1)

    # --- 预分配 (唯一的显存申请, 之后零分配) ---
    A = torch.randn(N, N, dtype=dtype, device=device)
    B = torch.randn(N, N, dtype=dtype, device=device)
    C = torch.empty(N, N, dtype=dtype, device=device)

    free_after = get_free_memory_gb(device)
    print(f"[AICore Filler] 分配完成, 剩余显存: {free_after:.2f} GB")
    print(f"[AICore Filler] 运行中... Ctrl+C 停止")
    print()

    # --- 信号处理 ---
    running = True

    def shutdown(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # --- 主循环 ---
    round_count = 0
    total_matmuls = 0
    paused = False
    t_start = time.time()
    t_last_report = t_start

    # 每轮 FLOPS: burst × 2N³
    flops_per_round = args.burst * 2.0 * (N ** 3)

    while running:
        # 定期检查显存
        if round_count % args.check_interval == 0:
            free_gb = get_free_memory_gb(device)
            if free_gb < args.min_free_gb:
                if not paused:
                    print(f"[AICore Filler] ⚠ 显存不足 ({free_gb:.2f}GB), 暂停计算...")
                    paused = True
                time.sleep(2.0)
                round_count += 1
                continue
            elif paused:
                print(f"[AICore Filler] ✓ 显存恢复 ({free_gb:.2f}GB), 恢复计算")
                paused = False

        # 一轮 burst: 连续 matmul, 写入预分配的 C (零分配)
        for _ in range(args.burst):
            torch.mm(A, B, out=C)
        total_matmuls += args.burst
        round_count += 1

        # 定期报告
        if round_count % args.report_interval == 0:
            t_now = time.time()
            dt = t_now - t_last_report
            if dt > 0:
                tflops = (args.report_interval * flops_per_round) / dt / 1e12
                free_gb = get_free_memory_gb(device)
                elapsed = t_now - t_start
                print(
                    f"[AICore Filler] round={round_count:,}, "
                    f"matmuls={total_matmuls:,}, "
                    f"~{tflops:.1f} TFLOPS, "
                    f"free={free_gb:.2f}GB, "
                    f"elapsed={elapsed:.0f}s"
                )
            t_last_report = t_now

        # 可控让步
        if args.sleep_ms > 0:
            time.sleep(args.sleep_ms / 1000.0)

        # 超时检查
        if args.max_seconds > 0 and (time.time() - t_start) >= args.max_seconds:
            print(f"[AICore Filler] 达到 max_seconds={args.max_seconds}, 退出")
            break

    # --- 清理 ---
    del A, B, C
    empty_cache()
    elapsed = time.time() - t_start
    print(f"\n[AICore Filler] 已停止. 共 {total_matmuls:,} 次 matmul, 运行 {elapsed:.1f}s")


if __name__ == "__main__":
    main()
