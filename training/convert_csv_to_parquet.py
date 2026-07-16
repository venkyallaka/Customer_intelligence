from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_CSV = r"C:\Users\vamsi\Downloads\Telegram Desktop\preprocessed_automobile_dataset.csv"


def convert(csv_path: str, parquet_path: str, chunksize: int) -> None:
    destination = Path(parquet_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()

    writer: pq.ParquetWriter | None = None
    rows = 0
    try:
        for chunk_index, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunksize, dtype=str, keep_default_na=False), start=1):
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(destination, table.schema, compression="zstd")
            else:
                table = table.cast(writer.schema)
            writer.write_table(table)
            rows += len(chunk)
            print(f"chunk={chunk_index} rows_written={rows}", flush=True)
    finally:
        if writer is not None:
            writer.close()
    print(f"done rows={rows} parquet={destination}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--parquet", default="data/preprocessed_automobile_dataset.parquet")
    parser.add_argument("--chunksize", type=int, default=75000)
    args = parser.parse_args()
    convert(args.csv, args.parquet, args.chunksize)


if __name__ == "__main__":
    main()
