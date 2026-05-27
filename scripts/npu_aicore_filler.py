#!/usr/bin/env python3
"""
NPU AICore 感知填充脚本 —— 通过 npu-smi 实时监测 AICore 利用率，
仅在利用率低于阈值时提交计算，训练活跃时自动退避。

架构:
  ┌──────────────────┐
  │  监控线程 (daemon) │  每 poll_interval 秒调一次 npu-smi info
  │  更新 cached %    │  解析目标卡的 AICore(%)
  └────────┬─────────┘
           │ 读 cached %
  ┌────────▼─────────┐
  │  主线程 决策循环   │  AICore < 阈值 → burst matmul (填空隙)
  │                   │  AICore >= 阈值 → sleep (让路给训练)
  └──────────────────┘

显存占用: 3 个 N×N 矩阵, 默认 1024×1024 float16 = ~6MB, 启动后零额外分配。

用法:
  # 单卡
  python npu_aicore_filler.py --device 0

  # 8 卡并行
  for i in 0 1 2 3 4 5 6 7; do
    python npu_aicore_filler.py --device $i &
  done

  # 保守模式 (更高阈值才触发, 更短 burst)
  python npu_aicore_filler.py --device 0 --aicore_threshold 15 --burst 30

  # 停止
  pkill -f npu_aicore_filler
"""

import os
import sys
import time
import signal
import argparse
import subprocess
import threading

import torch

try:
    import torch_npu
    HAS_NPU = True
except ImportError:
    HAS_NPU = False


# ─────────────────────────────────────────────
#  npu-smi 解析
# ─────────────────────────────────────────────

def parse_npu_smi_aicore(text, device_id):
    """从 npu-smi info 的文本输出中提取指定 device 的 AICore(%).

    npu-smi info 每张卡输出两行数据行 (夹在 === 分隔线之间):
        第1行: | 7     910B1       | OK     | Power  Temp  Hugepages |  ← NPU ID
        第2行: | 0                 | Bus-Id | 61     Mem   HBM       |  ← AICore%

    解析策略: 先在第1行匹配 device_id, 再从紧接的第2行提取 AICore%.
    """
    lines = text.splitlines()
    found_device = False
    for line in lines:
        parts = [p.strip() for p in line.split("|") if p.strip()]
        if len(parts) < 3:
            found_device = False
            continue

        first_tokens = parts[0].split()

        if not found_device:
            # 第1行: "7     910B1" → NPU ID = 7
            if len(first_tokens) >= 2:
                try:
                    npu_id = int(first_tokens[0])
                except ValueError:
                    continue
                if npu_id == device_id:
                    found_device = True
        else:
            # 第2行: parts[2] 的第一个 token 就是 AICore%
            third_tokens = parts[2].split()
            if third_tokens:
                try:
                    return int(third_tokens[0])
                except ValueError:
                    pass
            found_device = False
    return None


def poll_aicore_once(device_id):
    """调用 npu-smi info, 返回目标卡的 AICore% (int) 或 None."""
    try:
        r = subprocess.run(
            ["npu-smi", "info"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode != 0:
            return None
        return parse_npu_smi_aicore(r.stdout, device_id)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


# ─────────────────────────────────────────────
#  后台监控线程
# ─────────────────────────────────────────────

class AICoreMonitor:
    """Daemon 线程, 定期轮询 npu-smi 并缓存 AICore%.

    主线程通过 .aicore_pct 读取最近一次采样值 (无锁, int 赋值是原子的).
    """

    def __init__(self, device_id, poll_interval_s=0.5):
        self.device_id = device_id
        self.poll_interval_s = poll_interval_s
        self.aicore_pct = None          # None = 尚未拿到数据
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            val = poll_aicore_once(self.device_id)
            if val is not None:
                self.aicore_pct = val
            time.sleep(self.poll_interval_s)

    def stop(self):
        self._running = False
        self._thread.join(timeout=3)


# ─────────────────────────────────────────────
#  显存工具
# ─────────────────────────────────────────────

def get_free_memory_gb(device):
    if HAS_NPU:
        total = torch_npu.npu.get_device_properties(device).total_memory
        reserved = torch_npu.npu.memory_reserved(device)
    else:
        total = torch.cuda.get_device_properties(device).total_mem
        reserved = torch.cuda.memory_reserved(device)
    return (total - reserved) / (1024 ** 3)


def empty_cache():
    (torch_npu.npu if HAS_NPU else torch.cuda).empty_cache()


# ─────────────────────────────────────────────
#  参数
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="NPU AICore 感知填充: 仅在 AICore 空闲时计算, 训练活跃时自动退避",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    g_dev = p.add_argument_group("设备与精度")
    g_dev.add_argument("--device", type=int, default=0,
                       help="NPU device ID (默认 0)")
    g_dev.add_argument("--matrix_size", type=int, default=1024,
                       help="N×N matmul 维度. 1024 → ~6MB 显存")
    g_dev.add_argument("--dtype", type=str, default="float16",
                       choices=["float16", "bfloat16", "float32"])

    g_sense = p.add_argument_group("感知调度 (核心)")
    g_sense.add_argument("--aicore_threshold", type=int, default=45,
                         help="AICore%% 低于此值时才提交 matmul (默认 45). "
                              "训练 forward/backward 时通常 >60%%, 数据加载时 <10%%")
    g_sense.add_argument("--poll_interval", type=float, default=0.1,
                         help="npu-smi 轮询间隔 (秒). 加上 npu-smi 自身 ~100ms, "
                              "实际探测周期 ≈ poll_interval + 0.1s. 默认 0.1 → 周期 ~200ms")
    g_sense.add_argument("--burst", type=int, default=50,
                         help="每次填充的连续 matmul 次数. 50 次 ≈ 1-2ms, "
                              "短 burst 让主循环快速回到判断点")
    g_sense.add_argument("--fill_sleep_ms", type=float, default=1.0,
                         help="填充轮之间的 sleep (ms). 给训练 kernel 插队的窗口")
    g_sense.add_argument("--idle_sleep_ms", type=float, default=100.0,
                         help="AICore 繁忙时的 sleep (ms). 越大→越少干扰训练")

    g_safe = p.add_argument_group("安全保护")
    g_safe.add_argument("--min_free_gb", type=float, default=5.0,
                        help="设备剩余显存安全线 (GB). 低于则暂停计算")
    g_safe.add_argument("--max_usage_mb", type=float, default=1024.0,
                        help="脚本自身显存上限 (MB). 超过则拒绝启动")
    g_safe.add_argument("--mem_check_interval", type=int, default=200,
                        help="每隔多少轮检查一次显存")

    g_misc = p.add_argument_group("其他")
    g_misc.add_argument("--report_interval", type=int, default=1000,
                        help="每隔多少轮打印统计")
    g_misc.add_argument("--max_seconds", type=int, default=0,
                        help="最大运行秒数, 0=无限制")

    return p.parse_args()


# ─────────────────────────────────────────────
#  主流程
# ─────────────────────────────────────────────

def main():
    args = parse_args()

    device_type = "npu" if HAS_NPU else "cuda"
    device = torch.device(f"{device_type}:{args.device}")
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]
    bytes_per_elem = 2 if dtype in (torch.float16, torch.bfloat16) else 4
    N = args.matrix_size
    mem_mb = 3 * N * N * bytes_per_elem / (1024 ** 2)

    # ── 启动前校验 ──
    if HAS_NPU:
        torch_npu.npu.set_device(args.device)
    else:
        torch.cuda.set_device(args.device)

    if mem_mb > args.max_usage_mb:
        print(f"[Filler] 拒绝: 预计 {mem_mb:.1f}MB > max_usage_mb {args.max_usage_mb:.0f}MB")
        sys.exit(1)

    free_gb = get_free_memory_gb(device)
    if free_gb < args.min_free_gb:
        print(f"[Filler] 拒绝: 设备剩余 {free_gb:.2f}GB < min_free_gb {args.min_free_gb:.1f}GB")
        sys.exit(1)

    # ── npu-smi 连通性检查 ──
    test_reading = poll_aicore_once(args.device)
    if test_reading is None:
        print("[Filler] 错误: npu-smi info 无法获取 AICore%, 请确认 npu-smi 可用")
        print("[Filler] 尝试运行: npu-smi info")
        sys.exit(1)

    # ── 预分配 (唯一显存申请) ──
    A = torch.randn(N, N, dtype=dtype, device=device)
    B = torch.randn(N, N, dtype=dtype, device=device)
    C = torch.empty(N, N, dtype=dtype, device=device)
    free_after = get_free_memory_gb(device)

    print(f"[Filler] device={device}, dtype={args.dtype}, matrix={N}x{N}, 显存={mem_mb:.1f}MB")
    print(f"[Filler] 设备剩余: {free_after:.2f}GB (安全线 {args.min_free_gb:.1f}GB)")
    print(f"[Filler] AICore 阈值: <{args.aicore_threshold}% 时填充, >={args.aicore_threshold}% 时退避")
    print(f"[Filler] 当前 AICore: {test_reading}%")
    print(f"[Filler] burst={args.burst} (~{args.burst * 30 / 1000:.1f}ms), "
          f"poll={args.poll_interval}s, fill_sleep={args.fill_sleep_ms}ms, idle_sleep={args.idle_sleep_ms}ms")
    print()

    # ── 启动监控线程 ──
    monitor = AICoreMonitor(args.device, poll_interval_s=args.poll_interval)

    # ── 信号处理 ──
    running = True

    def shutdown(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # ── 主循环 ──
    round_count = 0
    total_matmuls = 0
    fill_rounds = 0
    idle_rounds = 0
    mem_paused = False
    t_start = time.time()
    t_last_report = t_start
    flops_per_burst = args.burst * 2.0 * (N ** 3)

    while running:
        # 显存安全检查 (低频)
        if round_count % args.mem_check_interval == 0:
            free_gb = get_free_memory_gb(device)
            if free_gb < args.min_free_gb:
                if not mem_paused:
                    print(f"[Filler] 显存不足 ({free_gb:.2f}GB), 暂停...")
                    mem_paused = True
                time.sleep(2.0)
                round_count += 1
                continue
            elif mem_paused:
                print(f"[Filler] 显存恢复 ({free_gb:.2f}GB), 继续")
                mem_paused = False

        # ── 核心决策: 读 cached AICore% ──
        aicore = monitor.aicore_pct

        if aicore is not None and aicore < args.aicore_threshold:
            # AICore 空闲 → 填充
            for _ in range(args.burst):
                torch.mm(A, B, out=C)
            total_matmuls += args.burst
            fill_rounds += 1
            if args.fill_sleep_ms > 0:
                time.sleep(args.fill_sleep_ms / 1000.0)
        else:
            # AICore 繁忙 或 数据未就绪 → 退避
            idle_rounds += 1
            time.sleep(args.idle_sleep_ms / 1000.0)

        round_count += 1

        # 定期报告
        if round_count % args.report_interval == 0:
            t_now = time.time()
            dt = t_now - t_last_report
            elapsed = t_now - t_start
            fill_tflops = (fill_rounds * flops_per_burst) / dt / 1e12 if dt > 0 else 0
            fill_pct = fill_rounds / max(1, fill_rounds + idle_rounds) * 100
            cur_aicore = monitor.aicore_pct
            cur_aicore_s = f"{cur_aicore}%" if cur_aicore is not None else "N/A"

            print(
                f"[Filler] round={round_count:,}  "
                f"AICore={cur_aicore_s}  "
                f"fill/idle={fill_rounds}/{idle_rounds} ({fill_pct:.0f}% fill)  "
                f"matmuls={total_matmuls:,}  "
                f"~{fill_tflops:.1f}TFLOPS  "
                f"elapsed={elapsed:.0f}s"
            )
            # 重置区间计数
            fill_rounds = 0
            idle_rounds = 0
            t_last_report = t_now

        # 超时
        if args.max_seconds > 0 and (time.time() - t_start) >= args.max_seconds:
            print(f"[Filler] 达到 max_seconds={args.max_seconds}, 退出")
            break

    # ── 清理 ──
    monitor.stop()
    del A, B, C
    empty_cache()
    elapsed = time.time() - t_start
    print(f"\n[Filler] 已停止. matmuls={total_matmuls:,}, 运行 {elapsed:.1f}s")


if __name__ == "__main__":
    main()
