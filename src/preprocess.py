"""
Data preprocessing module for the ML project.

Uses Polars Lazy to efficiently process partitioned Parquet data,
compute rolling features over previous years, and output featurized data.
"""

import logging
import re
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

import typer
import polars as pl
from polars import LazyFrame
import pyarrow.dataset as ds  # for partitioned write

logger = logging.getLogger(__name__)


# Import the config loader
from src.config import PreprocessConfig as ConfigPreprocessConfig


class DataPreprocessor:
    """Preprocesses patient-year data with rolling features using Polars Lazy."""

    def __init__(self, config: ConfigPreprocessConfig):
        self.config = config

        # Default numeric columns for feature engineering
        if self.config.numeric_columns is None:
            self.config.numeric_columns = [
                "age", "bmi", "systolic_bp", "diastolic_bp", "heart_rate",
                "temperature", "glucose", "cholesterol_total", "hdl", "ldl",
                "triglycerides", "creatinine", "hemoglobin", "white_blood_cells",
                "platelets", "num_encounters", "num_medications", "num_lab_tests"
            ]
            
        # Target variable for Alzheimer's prediction
        self.target_column = "alzheimers_diagnosis"

    def _get_partitioned_data(self) -> LazyFrame:
        """Read partitioned Parquet data using Polars Lazy.

        Supports both:  <input_dir>/year=2019/*.parquet  (Hive style)
                        <input_dir>/2019/*.parquet      (numeric dir)
        Ensures a 'year' column exists.
        """
        input_path = Path(self.config.input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory {input_path} does not exist")

        logger.info(f"Reading partitioned data from {input_path}")

        lazy_frames: List[LazyFrame] = []

        # First try Hive-style: year=YYYY
        for d in input_path.iterdir():
            if d.is_dir():
                m = re.match(r"^year=(\d{4})$", d.name)
                if m:
                    year = int(m.group(1))
                    lf = pl.scan_parquet(str(d / "*.parquet")).with_columns(pl.lit(year).alias("year"))
                    lazy_frames.append(lf)

        # Also accept plain numeric dirs: 2019/
        for d in input_path.iterdir():
            if d.is_dir() and d.name.isdigit():
                year = int(d.name)
                lf = pl.scan_parquet(str(d / "*.parquet")).with_columns(pl.lit(year).alias("year"))
                lazy_frames.append(lf)

        # Fallback: scan all parquet files recursively, assuming they ALREADY contain 'year'
        if not lazy_frames:
            glob = str(input_path / "**" / "*.parquet")
            logger.warning("No explicit year partitions found; scanning all parquet recursively "
                           f"and assuming a 'year' column exists. Pattern: {glob}")
            lf = pl.scan_parquet(glob, recursive=True)
            if "year" not in lf.columns:
                raise ValueError("No 'year' column found and no partition dirs like 'year=YYYY' present.")
            lazy_frames = [lf]

        df = pl.concat(lazy_frames, how="vertical")
        logger.info(f"Loaded partitioned data. Columns: {df.columns}")
        return df

    def _compute_rolling_features(self, df: LazyFrame) -> LazyFrame:
        """Compute rolling features over previous years for each patient."""
        logger.info(f"Computing rolling features with window={self.config.rolling_window_years}")
        df_sorted = df.sort(["patient_id", "year"])

        rolling_exprs = []
        for col in self.config.numeric_columns:
            rolling_exprs.extend([
                pl.col(col).rolling_mean(
                    window_size=self.config.rolling_window_years, min_periods=1
                ).over("patient_id").alias(f"{col}_rolling_mean"),

                pl.col(col).rolling_max(
                    window_size=self.config.rolling_window_years, min_periods=1
                ).over("patient_id").alias(f"{col}_rolling_max"),

                pl.col(col).rolling_sum(
                    window_size=self.config.rolling_window_years, min_periods=1
                ).over("patient_id").alias(f"{col}_rolling_sum"),

                # count of non-null observations in the window
                pl.col(col).is_not_null().cast(pl.Int8).rolling_sum(
                    window_size=self.config.rolling_window_years, min_periods=1
                ).over("patient_id").alias(f"{col}_rolling_count"),
            ])

        return df_sorted.with_columns(rolling_exprs)

    def _compute_delta_features(self, df: LazyFrame) -> LazyFrame:
        """Compute delta features (change from previous year) for numeric columns."""
        logger.info("Computing delta features (year-over-year changes)")
        df_sorted = df.sort(["patient_id", "year"])

        delta_exprs = []
        for col in self.config.numeric_columns:
            prev = pl.col(col).shift(1).over("patient_id")
            delta = (pl.col(col) - prev).alias(f"{col}_delta")

            pct = pl.when(prev != 0).then(
                (pl.col(col) - prev) / prev * 100
            ).otherwise(0).alias(f"{col}_pct_change")

            delta_exprs.extend([delta, pct])

        return df_sorted.with_columns(delta_exprs)

    def _compute_aggregate_features(self, df: LazyFrame) -> LazyFrame:
        """Compute aggregate features across all years for each patient (replicated per row)."""
        logger.info("Computing aggregate features per patient")
        agg_exprs = []
        for col in self.config.numeric_columns:
            agg_exprs.extend([
                pl.col(col).mean().over("patient_id").alias(f"{col}_patient_mean"),
                pl.col(col).std().over("patient_id").alias(f"{col}_patient_std"),
                pl.col(col).min().over("patient_id").alias(f"{col}_patient_min"),
                pl.col(col).max().over("patient_id").alias(f"{col}_patient_max"),
            ])
        return df.with_columns(agg_exprs)

    def _compute_risk_features(self, df: LazyFrame) -> LazyFrame:
        """Compute risk-based features and flags."""
        logger.info("Computing risk-based features")
        risk_exprs = [
            # BMI categories
            pl.when(pl.col("bmi") < 18.5).then(1).otherwise(0).alias("underweight"),
            pl.when((pl.col("bmi") >= 18.5) & (pl.col("bmi") < 25)).then(1).otherwise(0).alias("normal_weight"),
            pl.when((pl.col("bmi") >= 25) & (pl.col("bmi") < 30)).then(1).otherwise(0).alias("overweight"),
            pl.when(pl.col("bmi") >= 30).then(1).otherwise(0).alias("obese"),
            # Blood pressure categories
            pl.when(pl.col("systolic_bp") >= 140).then(1).otherwise(0).alias("high_systolic"),
            pl.when(pl.col("diastolic_bp") >= 90).then(1).otherwise(0).alias("high_diastolic"),
            # Glucose categories
            pl.when(pl.col("glucose") >= 126).then(1).otherwise(0).alias("high_glucose"),
            pl.when((pl.col("glucose") >= 100) & (pl.col("glucose") < 126)).then(1).otherwise(0).alias("prediabetic"),
            # Cholesterol categories
            pl.when(pl.col("cholesterol_total") >= 240).then(1).otherwise(0).alias("high_cholesterol"),
            pl.when(pl.col("hdl") < 40).then(1).otherwise(0).alias("low_hdl"),
            pl.when(pl.col("ldl") >= 160).then(1).otherwise(0).alias("high_ldl"),
            # Age categories
            pl.when(pl.col("age") >= 65).then(1).otherwise(0).alias("elderly"),
            pl.when((pl.col("age") >= 45) & (pl.col("age") < 65)).then(1).otherwise(0).alias("middle_aged"),
            # Simple composite risk score
            (pl.when(pl.col("bmi") >= 30).then(1).otherwise(0) +
             pl.when(pl.col("systolic_bp") >= 140).then(1).otherwise(0) +
             pl.when(pl.col("glucose") >= 126).then(1).otherwise(0) +
             pl.when(pl.col("age") >= 65).then(1).otherwise(0)).alias("risk_score"),
        ]
        return df.with_columns(risk_exprs)

    def _handle_missing_values(self, df: LazyFrame) -> LazyFrame:
        """Handle missing values in the dataset."""
        logger.info("Handling missing values")
        fill_exprs = []

        for col in self.config.numeric_columns:
            fill_exprs.extend([
                pl.col(f"{col}_rolling_mean").fill_null(0),
                pl.col(f"{col}_rolling_max").fill_null(0),
                pl.col(f"{col}_rolling_sum").fill_null(0),
                pl.col(f"{col}_rolling_count").fill_null(0),
                pl.col(f"{col}_delta").fill_null(0),
                pl.col(f"{col}_pct_change").fill_null(0),
            ])

        risk_cols = [
            "underweight", "normal_weight", "overweight", "obese",
            "high_systolic", "high_diastolic", "high_glucose", "prediabetic",
            "high_cholesterol", "low_hdl", "high_ldl", "elderly", "middle_aged", "risk_score"
        ]
        for col in risk_cols:
            fill_exprs.append(pl.col(col).fill_null(0))

        return df.with_columns(fill_exprs)

    def _write_partitioned_output(self, df: LazyFrame, output_path: Path) -> None:
        """Write featurized data to partitioned Parquet files (Hive style) using PyArrow."""
        logger.info(f"Writing featurized data to {output_path}")
        output_path.mkdir(parents=True, exist_ok=True)

        # Collect from LazyFrame → Arrow table
        arrow_tbl = df.collect().to_arrow()

        # Use pyarrow.dataset to write partitioned by 'year'
        ds.write_dataset(
            data=arrow_tbl,
            base_dir=str(output_path),
            format="parquet",
            partitioning=["year"],  # creates year=YYYY/ folders
            existing_data_behavior="overwrite_or_ignore",
        )
        logger.info("Featurized data written successfully")

    def _analyze_prevalence(self, df: LazyFrame) -> None:
        """Analyze the prevalence of Alzheimer's diagnosis in the dataset."""
        logger.info("Analyzing Alzheimer's diagnosis prevalence...")
        
        # Get overall statistics
        stats = df.select([
            pl.len().alias("total_rows"),
            pl.col(self.target_column).sum().alias("positive_cases")
        ]).collect()
        
        total_rows = stats[0]["total_rows"].item()
        positive_cases = stats[0]["positive_cases"].item()
        prevalence = positive_cases / total_rows if total_rows > 0 else 0
        
        logger.info(f"Total rows: {total_rows:,}")
        logger.info(f"Alzheimer's positive cases: {positive_cases:,}")
        logger.info(f"Prevalence: {prevalence:.1%}")
        
        # Analyze by year
        year_stats = df.group_by("year").agg([
            pl.len().alias("year_rows"),
            pl.col(self.target_column).sum().alias("year_positives")
        ]).collect()
        
        logger.info("Prevalence by year:")
        for row in year_stats.iter_rows():
            year, year_rows, year_positives = row
            year_prevalence = year_positives / year_rows if year_rows > 0 else 0
            logger.info(f"  {year}: {year_positives:,}/{year_rows:,} ({year_prevalence:.1%})")

    def preprocess(self) -> None:
        """Main preprocessing pipeline."""
        logger.info("Starting data preprocessing pipeline")

        df = self._get_partitioned_data()
        
        # Analyze prevalence before feature engineering
        self._analyze_prevalence(df)
        
        df = self._compute_rolling_features(df)
        df = self._compute_delta_features(df)
        df = self._compute_aggregate_features(df)
        df = self._compute_risk_features(df)
        df = self._handle_missing_values(df)

        output_path = Path(self.config.output_dir)
        self._write_partitioned_output(df, output_path)

        schema = df.schema
        logger.info(f"Preprocessing complete! Columns: {list(schema.keys())}")


def preprocess(
    config_file: str = typer.Option("config/preprocess.yaml", "--config", help="Configuration file path"),
    input_dir: Optional[str] = typer.Option(None, "--input-dir", help="Override input directory from config"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", help="Override output directory from config"),
    rolling_window_years: Optional[int] = typer.Option(None, "--rolling-window", help="Override rolling window from config"),
    chunk_size: Optional[int] = typer.Option(None, "--chunk-size", help="Override chunk size from config"),
    seed: Optional[int] = typer.Option(None, "--seed", help="Override seed from config"),
) -> None:
    """
    Preprocess patient-year data with rolling features using Polars Lazy.

    Example:
        python cli.py preprocess --config config/preprocess.yaml
        python cli.py preprocess --config config/preprocess.yaml --input-dir data/raw --output-dir data/featurized
    """
    # Load configuration from file
    from src.config import load_config
    config = load_config("preprocess", config_file)
    
    # Override config values if provided as command line arguments
    if input_dir is not None:
        config.input_dir = input_dir
    if output_dir is not None:
        config.output_dir = output_dir
    if rolling_window_years is not None:
        config.rolling_window_years = rolling_window_years
    if chunk_size is not None:
        config.chunk_size = chunk_size
    if seed is not None:
        config.seed = seed
    
    DataPreprocessor(config).preprocess()


if __name__ == "__main__":
    typer.run(preprocess)
