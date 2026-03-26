"""Small shared helpers for repeated Spark DataFrame configuration patterns."""

from __future__ import annotations

from typing import Sequence

from pyspark.sql import DataFrame, SparkSession


def configure_shuffle_partitions(
    spark: SparkSession,
    shuffle_partitions: int | None,
) -> None:
    if shuffle_partitions:
        spark.conf.set("spark.sql.shuffle.partitions", shuffle_partitions)


def verify_partition_columns(
    df: DataFrame,
    partition_by: list[str],
) -> None:
    if not partition_by:
        return

    unknown = [col for col in partition_by if col not in df.columns]
    if unknown:
        raise ValueError(
            "Unknown partition columns: "
            + ", ".join(unknown)
            + ". Available columns: "
            + ", ".join(df.columns)
        )


def repartition_dataframe(
    df: DataFrame,
    partitions: int | None,
    partition_by: list[str],
) -> DataFrame:
    if partitions:
        if partition_by:
            return df.repartition(partitions, *partition_by)
        return df.repartition(partitions)

    if partition_by:
        return df.repartition(*partition_by)

    return df


def build_spark_session(
    app_name: str,
    master: str | None,
) -> SparkSession:
    builder = SparkSession.builder.appName(app_name)
    if master:
        builder = builder.master(master)
    return builder.getOrCreate()


def write_partitioned_parquet(
    df: DataFrame,
    output_path: str,
    partition_by: Sequence[str],
    mode: str = "overwrite",
) -> None:
    writer = df.write.mode(mode)
    if partition_by:
        writer = writer.partitionBy(*partition_by)
    writer.parquet(output_path)
