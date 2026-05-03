import os
import math
import time
import psutil
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm  # 添加 tqdm 导入


def log_step(step_name, start_time):
    mem = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)  # in MB
    elapsed = time.time() - start_time
    print(f"[{step_name}] Elapsed time: {elapsed:.2f}s, Memory usage: {mem:.2f} MB")


schema = pa.schema([
    ("X", pa.list_(pa.float64())),
    ("soma_joinid", pa.int64()),
    ("dataset_id", pa.int64()),
    ("assay", pa.int64()),
    ("cell_type", pa.int64()),
    ("development_stage", pa.int64()),
    ("disease", pa.int64()),
    ("tissue", pa.int64()),
    ("sex", pa.int64()),
    ("tech_sample", pa.int64()),
    ("species", pa.int64()),
    ("idx", pa.int64()),
])
print("[Define schema] Done")


# 获取当前脚本的绝对路径
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
print(f"脚本路径: {script_path}")
print(f"脚本目录: {script_dir}")

input_path = f'{script_dir}/all_data/'
output_dir = f'{script_dir}/all_flatten_data/'
output_files = "all_flatten_part_{num}.parquet"
rows_per_file = 16384
print(f"数据路径: {input_path}")
print(f"输出目录: {output_dir}")

# read all parquet files in dir
start = time.time()
print("正在读取 Parquet 文件...")
df = pd.read_parquet(input_path, engine="pyarrow")
log_step("Load Parquet with Pandas", start)

start = time.time()
print("正在打乱数据...")
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
log_step("Shuffle rows", start)

# calculate final number of chunk files
start = time.time()
os.makedirs(output_dir, exist_ok=True)
total_rows = len(df)
num_files = math.ceil(total_rows / rows_per_file)
print(f"总行数: {total_rows}, 将生成 {num_files} 个文件")
log_step("Prepare to split and write", start)

# save chunk files
print("正在写入分块文件...")
for i in tqdm(range(num_files), desc="写入文件", unit="文件"):
    chunk_start = time.time()
    chunk = df.iloc[i * rows_per_file: (i + 1) * rows_per_file]
    table = pa.Table.from_pandas(chunk, schema=schema, preserve_index=False)
    output_path = os.path.join(output_dir, output_files.format(num=str(i)))
    pq.write_table(table, output_path, compression="snappy")
    # 如果你想保留每个文件的详细日志，可以取消下面这行的注释
    # log_step(f"Write file {output_files.format(num=str(i))}", chunk_start)

log_step("All done", start)
