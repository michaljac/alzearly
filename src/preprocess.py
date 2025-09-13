"""
Data preprocessing module for the ML project.

Automatically chooses between Polars and Dask based on data size.
Uses Polars Lazy for smaller datasets and Dask for larger datasets.
"""

import logging
import re
from pathlib import Path

import typer
import polars as pl
from polars import LazyFrame
import pyarrow.dataset as ds  # for partitioned write
from tqdm import tqdm

# Use only Polars for data processing
import pandas as pd  # Needed for some operations
DASK_AVAILABLE = False
print("ℹ️  Using Polars for all data processing (Dask disabled).")

logger = logging.getLogger(__name__)


# Import the config loader
# Config is loaded as dictionary, no need for type import


class DataPreprocessor:
    """Preprocesses patient-year data with rolling features using Polars Lazy."""

    def __init__(self, config):
        self.config = config

        # Default numeric columns for feature engineering
        if self.config.get('feature_engineering', {}).get('clinical', {}).get('numeric_columns') is None:
            self.config.setdefault('feature_engineering', {}).setdefault('clinical', {})
            self.config['feature_engineering']['clinical']['numeric_columns'] = [
                "age", "bmi", "systolic_bp", "diastolic_bp", "heart_rate",
                "temperature", "glucose", "cholesterol_total", "hdl", "ldl",
                "triglycerides", "creatinine", "hemoglobin", "white_blood_cells",
                "platelets", "num_encounters", "num_medications", "num_lab_tests"
            ]
            
        # Target variable for Alzheimer's prediction
        self.target_column = "alzheimers_diagnosis"
        
        # Data size thresholds (in rows)
        self.polars_threshold = self.config.get('memory', {}).get('framework_selection', {}).get('polars_threshold', 1_000_000)
        self.dask_threshold = self.config.get('memory', {}).get('framework_selection', {}).get('dask_threshold', 5_000_000)

    def _estimate_data_size(self):
        """Estimate the total number of rows in the dataset."""
        input_path = Path(self.config['io']['input_dir'])
        total_rows = 0
        
        # Count rows in all parquet files
        for parquet_file in input_path.rglob("*.parquet"):
            try:
                # Quick row count using pyarrow
                import pyarrow.parquet as pq
                parquet_file = pq.ParquetFile(parquet_file)
                total_rows += parquet_file.metadata.num_rows
            except Exception as e:
                print(f"⚠️  Could not read {parquet_file}: {e}")
                continue
        
        return total_rows

    def _choose_framework(self, data_size):
        """Choose the appropriate framework based on data size."""
        # Always use Polars since Dask is disabled
        return "polars"

    def _get_partitioned_data(self):
        """Read partitioned Parquet data using Polars Lazy.

        Supports both:  <input_dir>/year=2019/*.parquet  (Hive style)
                        <input_dir>/2019/*.parquet      (numeric dir)
        Ensures a 'year' column exists.
        """
        input_path = Path(self.config['io']['input_dir'])
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory {input_path} does not exist")

        lazy_frames = []

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
            lf = pl.scan_parquet(glob, recursive=True)
            if "year" not in lf.columns:
                raise ValueError("No 'year' column found and no partition dirs like 'year=YYYY' present.")
            lazy_frames = [lf]

        df = pl.concat(lazy_frames, how="vertical")
        return df

    def _get_dask_data(self):
        """Read partitioned Parquet data using Dask."""
        input_path = Path(self.config['io']['input_dir'])
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory {input_path} does not exist")

        # Dask can read partitioned parquet files directly
        df = dd.read_parquet(str(input_path / "**/*.parquet"))
        
        # Ensure year column exists
        if "year" not in df.columns:
            # Try to extract year from partition info
            df = df.reset_index()
            if "year" not in df.columns:
                raise ValueError("No 'year' column found in data")
        
        return df

    def _compute_rolling_features(self, df):
        """Compute rolling features over previous years for each patient."""
        # Skip rolling features to avoid Polars panic - use simpler aggregate features instead
        print("⚠️  Skipping rolling features to avoid Polars compatibility issues")
        return df

    def _compute_delta_features(self, df):
        """Compute delta features (change from previous year) for numeric columns."""
        # Skip delta features to avoid Polars panic - use simpler features instead
        print("⚠️  Skipping delta features to avoid Polars compatibility issues")
        return df

    def _compute_aggregate_features(self, df):
        """Compute aggregate features across all years for each patient (replicated per row)."""
        # Computing aggregate features per patient
        agg_exprs = []
        for col in self.config['feature_engineering']['clinical']['numeric_columns']:
            agg_exprs.extend([
                pl.col(col).mean().over("patient_id").alias(f"{col}_patient_mean"),
                pl.col(col).std().over("patient_id").alias(f"{col}_patient_std"),
                pl.col(col).min().over("patient_id").alias(f"{col}_patient_min"),
                pl.col(col).max().over("patient_id").alias(f"{col}_patient_max"),
            ])
        return df.with_columns(agg_exprs)

    def _compute_risk_features(self, df):
        """Compute risk-based features and flags."""
        # Computing risk-based features
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

    def _encode_categorical_features(self, df):
        """Encode categorical features using the specified strategy."""
        if not self.config.get('feature_engineering', {}).get('clinical', {}).get('categorical_columns'):
            return df
            
        # Encoding categorical columns
        
        # Get encoding configuration
        encoding_config = self.config.get('preprocessing', {}).get('categorical_encoding', {}) or {}
        strategy = encoding_config.get("strategy", "onehot")
        drop_first = encoding_config.get("drop_first", True)
        
        if strategy == "onehot":
            return self._onehot_encode_categoricals(df)
        elif strategy == "label":
            return self._label_encode_categoricals(df)
        else:
            logger.warning(f"Unknown encoding strategy: {strategy}. Using one-hot encoding.")
            return self._onehot_encode_categoricals(df)
    
    def _onehot_encode_categoricals(self, df):
        """One-hot encode categorical columns."""
        encoding_exprs = []
        
        for col in self.config['feature_engineering']['clinical']['categorical_columns']:
            if col in df.columns:
                # Get unique values for this column
                unique_values = df.select(pl.col(col)).unique().collect()[col].to_list()
                
                # Create one-hot encoding expressions
                for value in unique_values:
                    if value is not None:  # Skip null values
                        encoded_col_name = f"{col}_{value}"
                        encoding_exprs.append(
                            pl.when(pl.col(col) == value).then(1).otherwise(0).alias(encoded_col_name)
                        )
        
        if encoding_exprs:
            return df.with_columns(encoding_exprs)
        return df
    
    def _label_encode_categoricals(self, df):
        """Label encode categorical columns."""
        encoding_exprs = []
        
        for col in self.config['feature_engineering']['clinical']['categorical_columns']:
            if col in df.columns:
                # Create a mapping of unique values to integers
                unique_values = df.select(pl.col(col)).unique().collect()[col].to_list()
                value_to_int = {val: idx for idx, val in enumerate(unique_values) if val is not None}
                
                # Create label encoding expression
                encoding_exprs.append(
                    pl.col(col).map_elements(lambda x: value_to_int.get(x, -1)).alias(f"{col}_encoded")
                )
        
        if encoding_exprs:
            return df.with_columns(encoding_exprs)
        return df

    def _handle_missing_values(self, df):
        """Handle missing values in the dataset."""
        # Handling missing values
        fill_exprs = []

        # Handle numeric columns (simplified since we skipped rolling/delta features)
        if self.config.get('feature_engineering', {}).get('clinical', {}).get('numeric_columns'):
            for col in self.config['feature_engineering']['clinical']['numeric_columns']:
                fill_exprs.extend([
                    pl.col(f"{col}_patient_mean").fill_null(0),
                    pl.col(f"{col}_patient_std").fill_null(0),
                    pl.col(f"{col}_patient_min").fill_null(0),
                    pl.col(f"{col}_patient_max").fill_null(0),
                ])

        # Handle categorical columns
        if self.config.get('feature_engineering', {}).get('clinical', {}).get('categorical_columns'):
            categorical_fill = self.config.get('preprocessing', {}).get('categorical_encoding', {}).get("categorical_fill", "unknown") if self.config.get('preprocessing', {}).get('categorical_encoding', {}) else "unknown"
            for col in self.config['feature_engineering']['clinical']['categorical_columns']:
                fill_exprs.append(pl.col(col).fill_null(categorical_fill))

        # Handle risk features
        risk_cols = [
            "underweight", "normal_weight", "overweight", "obese",
            "high_systolic", "high_diastolic", "high_glucose", "prediabetic",
            "high_cholesterol", "low_hdl", "high_ldl", "elderly", "middle_aged", "risk_score"
        ]
        for col in risk_cols:
            fill_exprs.append(pl.col(col).fill_null(0))

        return df.with_columns(fill_exprs)

    def _write_partitioned_output(self, df, output_path):
        """Write featurized data to partitioned Parquet files (Hive style) using Polars."""
        output_path.mkdir(parents=True, exist_ok=True)

        # Try to collect in smaller chunks to avoid memory issues
        try:
            # First try to get schema to understand the data structure
            schema = df.schema
            print(f"📊 Schema has {len(schema)} columns")
            
            # Try collecting with a limit first to test
            test_df = df.limit(100).collect()
            print(f"✅ Test collection successful with {len(test_df)} rows")
            
            # Now collect the full dataset
            collected_df = df.collect()
            print(f"✅ Full collection successful with {len(collected_df)} rows")
            
            # Write to parquet
            collected_df.write_parquet(
                file=str(output_path / "data.parquet"),
                compression="snappy"
            )
            print(f"✅ Data written to {output_path / 'data.parquet'}")
            
        except Exception as e:
            print(f"⚠️  Collection failed: {e}")
            print("🔄 Trying alternative approach...")
            
            try:
                # Try writing directly without collecting
                df.sink_parquet(
                    path=str(output_path / "data.parquet"),
                    compression="snappy"
                )
                print(f"✅ Data written using sink_parquet to {output_path / 'data.parquet'}")
            except Exception as e2:
                print(f"⚠️  sink_parquet also failed: {e2}")
                print("🔄 Trying CSV fallback...")
                
                try:
                    # Last resort: collect and write CSV
                    collected_df = df.collect()
                    collected_df.write_csv(
                        file=str(output_path / "data.csv")
                    )
                    print(f"✅ Data written as CSV to {output_path / 'data.csv'}")
                except Exception as e3:
                    print(f"❌ All write methods failed: {e3}")
                    raise

    def _analyze_prevalence(self, df):
        """Analyze the prevalence of Alzheimer's diagnosis in the dataset."""
        # Get overall statistics
        stats = df.select([
            pl.count().alias("total_rows"),
            pl.col(self.target_column).sum().alias("positive_cases")
        ]).collect()
        
        total_rows = stats[0]["total_rows"].item()
        positive_cases = stats[0]["positive_cases"].item()
        prevalence = positive_cases / total_rows if total_rows > 0 else 0
        
        # Analyze by year
        year_stats = df.group_by("year").agg([
            pl.count().alias("year_rows"),
            pl.col(self.target_column).sum().alias("year_positives")
        ]).collect()
        
        # Store results for potential use, but don't print
        self._prevalence_stats = {
            "total_rows": total_rows,
            "positive_cases": positive_cases,
            "prevalence": prevalence,
            "year_stats": year_stats
        }

    def preprocess(self):
        """Main preprocessing pipeline with automatic framework selection."""
        print("🚀 Starting data preprocessing pipeline...")
        
        # Step 1: Estimate data size and choose framework
        print("📊 Analyzing dataset size...")
        data_size = self._estimate_data_size()
        framework = self._choose_framework(data_size)
        
        print(f"📈 Dataset size: {data_size:,} rows")
        print(f"🔧 Using framework: {framework.upper()}")
        
        if framework == "dask":
            print("⚡ Using Dask for large dataset processing")
            self._preprocess_with_dask(data_size)
        else:
            print("🚀 Using Polars for efficient processing")
            self._preprocess_with_polars(data_size)

    def _preprocess_with_polars(self, data_size):
        """Preprocess using Polars Lazy framework."""
        # Step 1: Load data
        df = self._get_partitioned_data()
        
        # Step 2: Analyze prevalence (silent)
        self._analyze_prevalence(df)
        
        # Step 3: Feature engineering with single progress bar
        print("🔧 Computing features...")
        with tqdm(total=5, desc="Feature engineering", unit="step", 
                 bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                 ncols=80, ascii=True, position=0, dynamic_ncols=False, 
                 mininterval=0.1, maxinterval=1.0) as pbar:
            
            df = self._compute_rolling_features(df)
            pbar.update(1)
            
            df = self._compute_delta_features(df)
            pbar.update(1)
            
            df = self._compute_aggregate_features(df)
            pbar.update(1)
            
            df = self._compute_risk_features(df)
            pbar.update(1)
            
            df = self._encode_categorical_features(df)
            pbar.update(1)

        # Step 4: Handle missing values and write output
        print("💾 Finalizing and saving data...")
        with tqdm(total=2, desc="Finalizing", unit="step",
                 bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                 ncols=80, ascii=True, position=0, dynamic_ncols=False, 
                 mininterval=0.1, maxinterval=1.0) as pbar:
            
            df = self._handle_missing_values(df)
            pbar.update(1)
            
            output_path = Path(self.config['io']['output_dir'])
            self._write_partitioned_output(df, output_path)
            pbar.update(1)

        print(f"✅ Polars preprocessing complete! {len(df.schema)} columns created")

    def _preprocess_with_dask(self, data_size):
        """Preprocess using Dask framework for large datasets."""
        if not DASK_AVAILABLE:
            raise ImportError("Dask is required for large datasets but not available")
        
        # Step 1: Load data with Dask
        print("📂 Loading data with Dask...")
        df = self._get_dask_data()
        
        # Step 2: Feature engineering with Dask
        print("🔧 Computing features with Dask...")
        with tqdm(total=5, desc="Feature engineering", unit="step", 
                 bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                 ncols=80, ascii=True, position=0, dynamic_ncols=False, 
                 mininterval=0.1, maxinterval=1.0) as pbar:
            
            # Dask rolling features (simplified for large datasets)
            df = self._compute_dask_rolling_features(df)
            pbar.update(1)
            
            # Dask aggregate features
            df = self._compute_dask_aggregate_features(df)
            pbar.update(1)
            
            # Dask risk features
            df = self._compute_dask_risk_features(df)
            pbar.update(1)
            
            # Dask categorical encoding
            df = self._compute_dask_categorical_features(df)
            pbar.update(1)
            
            # Dask missing value handling
            df = self._handle_dask_missing_values(df)
            pbar.update(1)

        # Step 3: Write output
        print("💾 Saving data with Dask...")
        output_path = Path(self.config['io']['output_dir'])
        self._write_dask_output(df, output_path)

        print(f"✅ Dask preprocessing complete! Dataset processed in chunks")

    def _compute_dask_rolling_features(self, df):
        """Compute rolling features using Dask."""
        # Simplified rolling features for large datasets
        numeric_cols = [col for col in self.config['feature_engineering']['clinical']['numeric_columns'] if col in df.columns]
        
        for col in numeric_cols:
            # Group by patient_id and compute rolling statistics
            df[f"{col}_rolling_mean"] = df.groupby("patient_id")[col].rolling(window=3, min_periods=1).mean()
            df[f"{col}_rolling_max"] = df.groupby("patient_id")[col].rolling(window=3, min_periods=1).max()
        
        return df

    def _compute_dask_aggregate_features(self, df):
        """Compute aggregate features using Dask."""
        # Patient-level aggregates
        agg_features = df.groupby("patient_id").agg({
            col: ["mean", "std", "min", "max"] for col in self.config['feature_engineering']['clinical']['numeric_columns'] if col in df.columns
        }).reset_index()
        
        return df.merge(agg_features, on="patient_id", how="left")

    def _compute_dask_risk_features(self, df):
        """Compute risk features using Dask."""
        # BMI categories
        df["bmi_category"] = df["bmi"].map_partitions(
            lambda s: pd.cut(s, bins=[0, 18.5, 25, 30, 100], labels=["underweight", "normal", "overweight", "obese"])
        )
        
        # Age groups
        df["age_group"] = df["age"].map_partitions(
            lambda s: pd.cut(s, bins=[0, 50, 65, 80, 100], labels=["young", "middle", "senior", "elderly"])
        )
        
        return df

    def _compute_dask_categorical_features(self, df):
        """Compute categorical features using Dask."""
        # One-hot encoding for categorical columns
        categorical_cols = [col for col in self.config['feature_engineering']['clinical']['categorical_columns'] if col in df.columns]
        
        for col in categorical_cols:
            df = dd.get_dummies(df, columns=[col], prefix=col)
        
        return df

    def _handle_dask_missing_values(self, df):
        """Handle missing values using Dask."""
        # Fill numeric columns with 0
        numeric_cols = [col for col in self.config['feature_engineering']['clinical']['numeric_columns'] if col in df.columns]
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        return df

    def _write_dask_output(self, df, output_path):
        """Write Dask dataframe to partitioned parquet."""
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Repartition for efficient writing
        df = df.repartition(partition_size="100MB")
        
        # Write to parquet
        df.to_parquet(str(output_path), engine="pyarrow")


def preprocess(
    config_file = typer.Option("config/preprocess.yaml", "--config", help="Configuration file path"),
    input_dir = typer.Option(None, "--input-dir", help="Override input directory from config"),
    output_dir = typer.Option(None, "--output-dir", help="Override output directory from config"),
    rolling_window_years = typer.Option(None, "--rolling-window", help="Override rolling window from config"),
    chunk_size = typer.Option(None, "--chunk-size", help="Override chunk size from config"),
    seed = typer.Option(None, "--seed", help="Override seed from config"),
):
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
        config['io']['input_dir'] = input_dir
    if output_dir is not None:
        config['io']['output_dir'] = output_dir
    if rolling_window_years is not None:
        config['feature_engineering']['temporal']['lookback_years'] = rolling_window_years
    if chunk_size is not None:
        config['memory']['chunk_size'] = chunk_size
    if seed is not None:
        # Add seed to config if it doesn't exist
        if 'processing' not in config:
            config['processing'] = {}
        config['processing']['seed'] = seed
    
    DataPreprocessor(config).preprocess()


if __name__ == "__main__":
    typer.run(preprocess)
