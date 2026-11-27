"""Disk-backed bucket and shard writers for the K1369 within-species shuffle."""
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def log(message):
    print(message, flush=True)


class BucketWriterPool:
    def __init__(self, directory, schema, count, buffer_rows, compression):
        self.directory = Path(directory)
        self.schema = schema
        self.buffer_rows = buffer_rows
        self.compression = compression
        self.bucket_sizes = np.zeros(count, dtype=np.int64)
        self.pending = [[] for _ in range(count)]
        self.pending_rows = np.zeros(count, dtype=np.int64)
        self.writers = {}

    def path(self, bucket_id):
        return self.directory / f'bucket_{bucket_id:06d}.parquet'

    def append(self, bucket_id, table):
        self.pending[bucket_id].append(table)
        self.bucket_sizes[bucket_id] += table.num_rows
        self.pending_rows[bucket_id] += table.num_rows
        if self.pending_rows[bucket_id] >= self.buffer_rows:
            self.flush(bucket_id)

    def flush(self, bucket_id):
        tables = self.pending[bucket_id]
        if not tables:
            return
        if bucket_id not in self.writers:
            self.writers[bucket_id] = pq.ParquetWriter(
                self.path(bucket_id), self.schema, compression=self.compression)
        self.writers[bucket_id].write_table(pa.concat_tables(tables))
        tables.clear()
        self.pending_rows[bucket_id] = 0

    def close(self):
        try:
            for bucket_id in range(len(self.pending)):
                self.flush(bucket_id)
        finally:
            for writer in self.writers.values():
                writer.close()


def split_into_buckets(table, ids, pool):
    order = np.argsort(ids, kind='stable')
    ordered_ids = ids[order]
    for indices in np.split(order, np.flatnonzero(np.diff(ordered_ids)) + 1):
        if indices.size:
            pool.append(int(ids[indices[0]]), table.take(pa.array(indices)))


class FixedSizeOutputWriter:
    def __init__(self, directory, rows_per_file, compression):
        self.directory = Path(directory)
        self.limit = rows_per_file
        self.compression = compression
        self.rows_written = 0
        self.writer = None

    def append(self, table):
        offset = 0
        while offset < table.num_rows:
            if self.writer is None:
                index = self.rows_written // self.limit
                self.writer = pq.ParquetWriter(
                    self.directory / f'shuffled_part_{index:08d}.parquet',
                    table.schema, compression=self.compression)
            count = min(self.limit - self.rows_written % self.limit, table.num_rows - offset)
            self.writer.write_table(table.slice(offset, count))
            offset += count
            self.rows_written += count
            if self.rows_written % self.limit == 0:
                self.close()

    def close(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None
