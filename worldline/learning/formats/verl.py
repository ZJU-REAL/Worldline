"""Bounded-memory Parquet output for external verl trainers."""


class ParquetStream:
    def __init__(self, path, schema, batch_size=128):
        import pyarrow.parquet as pq

        self.path, self.schema, self.batch_size, self.rows = path, schema, batch_size, []
        self.writer = pq.ParquetWriter(path, schema)

    def append(self, row):
        self.rows.append(row)
        if len(self.rows) >= self.batch_size:
            self.flush()

    def flush(self):
        import pyarrow as pa

        if self.rows:
            self.writer.write_table(pa.Table.from_pylist(self.rows, schema=self.schema))
            self.rows.clear()

    def close(self):
        self.flush()
        self.writer.close()
