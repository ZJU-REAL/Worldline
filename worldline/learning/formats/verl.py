"""Byte-bounded Parquet output for external verl trainers."""


def record_bytes(value):
    if isinstance(value, bytes):
        return len(value)
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    if isinstance(value, dict):
        return sum(record_bytes(key) + record_bytes(nested) for key, nested in value.items())
    if isinstance(value, (tuple, list)):
        return sum(record_bytes(nested) for nested in value)
    return 16


class ParquetStream:
    def __init__(
        self, path, schema, batch_size=128, *, max_batch_bytes=8 * 1024 * 1024, max_record_bytes=64 * 1024 * 1024
    ):
        import pyarrow.parquet as pq

        if min(batch_size, max_batch_bytes, max_record_bytes) < 1:
            raise ValueError("Parquet buffer limits must be positive")
        self.path, self.schema, self.batch_size, self.rows = path, schema, batch_size, []
        self.max_batch_bytes, self.max_record_bytes = max_batch_bytes, max_record_bytes
        self.buffered_bytes = self.peak_buffered_bytes = self.row_groups = 0
        self.writer = pq.ParquetWriter(path, schema)

    def prepare(self, row):
        import pyarrow as pa

        size = record_bytes(row)
        if size > self.max_record_bytes:
            raise ValueError("Parquet record exceeds configured byte budget")
        table = pa.Table.from_pylist([row], schema=self.schema)
        size = max(size, table.nbytes)
        if size > self.max_record_bytes:
            raise ValueError("Parquet record exceeds configured byte budget")
        return table, size

    def append(self, row):
        self.append_prepared(self.prepare(row))

    def append_prepared(self, prepared):
        table, size = prepared
        if self.rows and self.buffered_bytes + size > self.max_batch_bytes:
            self.flush()
        self.rows.append(table)
        self.buffered_bytes += size
        self.peak_buffered_bytes = max(self.peak_buffered_bytes, self.buffered_bytes)
        if len(self.rows) >= self.batch_size or self.buffered_bytes >= self.max_batch_bytes:
            self.flush()

    def flush(self):
        import pyarrow as pa

        if self.rows:
            self.writer.write_table(pa.concat_tables(self.rows))
            self.rows.clear()
            self.buffered_bytes = 0
            self.row_groups += 1

    def close(self):
        self.flush()
        self.writer.close()
