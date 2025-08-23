"""
Synthetic data generation module for patient-year datasets.

Generates realistic patient data with demographic, clinical, and temporal features.
Supports chunked generation to handle large datasets efficiently.
"""

import logging
import random
from pathlib import Path
from typing import List, Optional, Dict, Any, Iterator
from dataclasses import dataclass

import typer
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
from faker import Faker

logger = logging.getLogger(__name__)


@dataclass
class DataGenConfig:
    """Configuration for synthetic data generation."""
    n_patients: int
    years: List[int]
    positive_rate: float = 0.07
    rows_per_chunk: int = 100_000
    seed: int = 0
    output_dir: str = "data/raw"


class SyntheticDataGenerator:
    """Generates synthetic patient-year data with realistic features."""
    
    def __init__(self, config: DataGenConfig):
        self.config = config
        self.fake = Faker()
        Faker.seed(config.seed)
        random.seed(config.seed)
        np.random.seed(config.seed)
        
        # Initialize patient states for consistent generation
        self.patient_states = {}
        
    def _create_schema(self) -> pa.Schema:
        """Create PyArrow schema for the dataset."""
        return pa.schema([
            # Patient identifiers
            pa.field("patient_id", pa.string()),
            pa.field("year", pa.int32()),
            
            # Demographics (categorical)
            pa.field("sex", pa.string()),
            pa.field("region", pa.string()),
            pa.field("occupation", pa.string()),
            pa.field("education_level", pa.string()),
            pa.field("marital_status", pa.string()),
            pa.field("insurance_type", pa.string()),
            
            # Clinical features (numeric)
            pa.field("age", pa.float32()),
            pa.field("bmi", pa.float32()),
            pa.field("systolic_bp", pa.float32()),
            pa.field("diastolic_bp", pa.float32()),
            pa.field("heart_rate", pa.float32()),
            pa.field("temperature", pa.float32()),
            pa.field("glucose", pa.float32()),
            pa.field("cholesterol_total", pa.float32()),
            pa.field("hdl", pa.float32()),
            pa.field("ldl", pa.float32()),
            pa.field("triglycerides", pa.float32()),
            pa.field("creatinine", pa.float32()),
            pa.field("hemoglobin", pa.float32()),
            pa.field("white_blood_cells", pa.float32()),
            pa.field("platelets", pa.float32()),
            
            # Encounter features
            pa.field("num_encounters", pa.int32()),
            pa.field("num_medications", pa.int32()),
            pa.field("num_lab_tests", pa.int32()),
            
            # Target variable
            pa.field("diagnosis", pa.int8()),
        ])
    
    def _initialize_patient_state(self, patient_id: str) -> Dict[str, Any]:
        """Initialize or retrieve patient state for consistent generation."""
        if patient_id not in self.patient_states:
            # Generate stable demographic features
            self.patient_states[patient_id] = {
                "sex": self.fake.random_element(["M", "F"]),
                "region": self.fake.state(),
                "occupation": self.fake.job(),
                "education_level": self.fake.random_element(
                    ["High School", "Bachelor's", "Master's", "PhD", "Some College"]
                ),
                "marital_status": self.fake.random_element(
                    ["Single", "Married", "Divorced", "Widowed"]
                ),
                "insurance_type": self.fake.random_element(
                    ["Private", "Medicare", "Medicaid", "Uninsured"]
                ),
                "base_age": self.fake.random_int(min=18, max=85),
                "base_bmi": self.fake.pyfloat(min_value=18.5, max_value=40.0),
                "base_systolic": self.fake.pyfloat(min_value=90, max_value=180),
                "base_diastolic": self.fake.pyfloat(min_value=60, max_value=110),
                "base_heart_rate": self.fake.pyfloat(min_value=60, max_value=100),
                "base_temperature": self.fake.pyfloat(min_value=97.0, max_value=99.5),
                "base_glucose": self.fake.pyfloat(min_value=70, max_value=140),
                "base_cholesterol": self.fake.pyfloat(min_value=150, max_value=300),
                "base_hdl": self.fake.pyfloat(min_value=30, max_value=80),
                "base_ldl": self.fake.pyfloat(min_value=70, max_value=200),
                "base_triglycerides": self.fake.pyfloat(min_value=50, max_value=400),
                "base_creatinine": self.fake.pyfloat(min_value=0.6, max_value=1.4),
                "base_hemoglobin": self.fake.pyfloat(min_value=12, max_value=18),
                "base_wbc": self.fake.pyfloat(min_value=4, max_value=11),
                "base_platelets": self.fake.pyfloat(min_value=150, max_value=450),
                
                # Random walk states for temporal trends
                "bmi_walk": 0.0,
                "bp_walk": 0.0,
                "glucose_walk": 0.0,
                "cholesterol_walk": 0.0,
                "creatinine_walk": 0.0,
                
                # Diagnosis probability (increases with age and risk factors)
                "diagnosis_prob": 0.0,
            }
        return self.patient_states[patient_id]
    
    def _generate_patient_year_data(self, patient_id: str, year: int) -> Dict[str, Any]:
        """Generate data for a single patient-year combination."""
        state = self._initialize_patient_state(patient_id)
        
        # Calculate age progression
        age = state["base_age"] + (year - min(self.config.years))
        
        # Update random walks for temporal trends
        state["bmi_walk"] += np.random.normal(0, 0.5)
        state["bp_walk"] += np.random.normal(0, 2.0)
        state["glucose_walk"] += np.random.normal(0, 3.0)
        state["cholesterol_walk"] += np.random.normal(0, 5.0)
        state["creatinine_walk"] += np.random.normal(0, 0.05)
        
        # Generate clinical values with realistic ranges and trends
        bmi = max(16.0, min(50.0, state["base_bmi"] + state["bmi_walk"]))
        systolic = max(70, min(220, state["base_systolic"] + state["bp_walk"]))
        diastolic = max(40, min(130, systolic * 0.6 + np.random.normal(0, 5)))
        heart_rate = max(50, min(120, state["base_heart_rate"] + np.random.normal(0, 5)))
        temperature = max(95.0, min(103.0, state["base_temperature"] + np.random.normal(0, 0.5)))
        glucose = max(50, min(300, state["base_glucose"] + state["glucose_walk"]))
        cholesterol_total = max(100, min(500, state["base_cholesterol"] + state["cholesterol_walk"]))
        hdl = max(20, min(100, state["base_hdl"] + np.random.normal(0, 3)))
        ldl = max(30, min(250, cholesterol_total - hdl - np.random.normal(50, 10)))
        triglycerides = max(30, min(600, state["base_triglycerides"] + np.random.normal(0, 20)))
        creatinine = max(0.3, min(3.0, state["base_creatinine"] + state["creatinine_walk"]))
        hemoglobin = max(8, min(22, state["base_hemoglobin"] + np.random.normal(0, 0.5)))
        wbc = max(2, min(20, state["base_wbc"] + np.random.normal(0, 1)))
        platelets = max(50, min(600, state["base_platelets"] + np.random.normal(0, 30)))
        
        # Generate encounter counts (correlated with age and health status)
        health_risk = (age - 40) / 40 + (bmi - 25) / 10 + (glucose - 100) / 50
        # Ensure positive lambda values for Poisson distribution
        encounter_lambda = max(0.1, 2 + health_risk * 3)
        medication_lambda = max(0.1, 1 + health_risk * 2)
        lab_lambda = max(0.1, 3 + health_risk * 4)
        
        num_encounters = max(0, int(np.random.poisson(encounter_lambda)))
        num_medications = max(0, int(np.random.poisson(medication_lambda)))
        num_lab_tests = max(0, int(np.random.poisson(lab_lambda)))
        
        # Calculate diagnosis probability based on risk factors
        risk_factors = [
            age > 65,
            bmi > 30,
            systolic > 140,
            glucose > 126,
            cholesterol_total > 240,
            creatinine > 1.2,
        ]
        risk_score = sum(risk_factors) / len(risk_factors)
        state["diagnosis_prob"] = min(0.95, 0.05 + risk_score * 0.3)
        
        # Generate diagnosis
        diagnosis = 1 if np.random.random() < state["diagnosis_prob"] else 0
        
        return {
            "patient_id": patient_id,
            "year": year,
            "sex": state["sex"],
            "region": state["region"],
            "occupation": state["occupation"],
            "education_level": state["education_level"],
            "marital_status": state["marital_status"],
            "insurance_type": state["insurance_type"],
            "age": float(age),
            "bmi": float(bmi),
            "systolic_bp": float(systolic),
            "diastolic_bp": float(diastolic),
            "heart_rate": float(heart_rate),
            "temperature": float(temperature),
            "glucose": float(glucose),
            "cholesterol_total": float(cholesterol_total),
            "hdl": float(hdl),
            "ldl": float(ldl),
            "triglycerides": float(triglycerides),
            "creatinine": float(creatinine),
            "hemoglobin": float(hemoglobin),
            "white_blood_cells": float(wbc),
            "platelets": float(platelets),
            "num_encounters": int(num_encounters),
            "num_medications": int(num_medications),
            "num_lab_tests": int(num_lab_tests),
            "diagnosis": int(diagnosis),
        }
    
    def _generate_chunk(self, chunk_size: int) -> Iterator[Dict[str, Any]]:
        """Generate a chunk of patient-year data."""
        patients_per_chunk = chunk_size // len(self.config.years)
        
        for i in range(0, self.config.n_patients, patients_per_chunk):
            chunk_patients = min(patients_per_chunk, self.config.n_patients - i)
            
            for j in range(chunk_patients):
                patient_id = f"P{i + j:06d}"
                
                for year in self.config.years:
                    yield self._generate_patient_year_data(patient_id, year)
    
    def _write_chunk_to_parquet(self, chunk_data: List[Dict[str, Any]], output_path: Path):
        """Write a chunk of data to Parquet with year partitioning."""
        if not chunk_data:
            return
            
        # Convert to pandas DataFrame first, then to PyArrow table
        df = pd.DataFrame(chunk_data)
        table = pa.Table.from_pandas(df)
        
        # Write to partitioned dataset
        ds.write_dataset(
            table,
            output_path,
            format="parquet",
            partitioning=ds.partitioning(pa.schema([("year", pa.int32())])),
            existing_data_behavior="overwrite_or_ignore",
        )
    
    def generate(self) -> None:
        """Generate synthetic data and save to partitioned Parquet files."""
        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Generating synthetic data for {self.config.n_patients} patients "
                   f"across years {self.config.years}")
        logger.info(f"Target positive rate: {self.config.positive_rate:.1%}")
        logger.info(f"Chunk size: {self.config.rows_per_chunk:,} rows")
        
        chunk_data = []
        total_rows = 0
        
        for row in self._generate_chunk(self.config.rows_per_chunk):
            chunk_data.append(row)
            total_rows += 1
            
            if len(chunk_data) >= self.config.rows_per_chunk:
                self._write_chunk_to_parquet(chunk_data, output_path)
                logger.info(f"Written chunk with {len(chunk_data):,} rows "
                           f"(total: {total_rows:,})")
                chunk_data = []
        
        # Write remaining data
        if chunk_data:
            self._write_chunk_to_parquet(chunk_data, output_path)
            logger.info(f"Written final chunk with {len(chunk_data):,} rows "
                       f"(total: {total_rows:,})")
        
        # Calculate actual positive rate
        actual_positive_rate = sum(1 for row in self._generate_chunk(total_rows) 
                                 if row["diagnosis"] == 1) / total_rows
        
        logger.info(f"Data generation complete!")
        logger.info(f"Total rows generated: {total_rows:,}")
        logger.info(f"Actual positive rate: {actual_positive_rate:.1%}")
        logger.info(f"Output directory: {output_path.absolute()}")


def generate(
    n_patients: int = typer.Option(..., "--n-patients", help="Number of patients to generate"),
    years: List[int] = typer.Option(..., "--years", help="Years to generate data for"),
    rows: Optional[int] = typer.Option(None, "--rows", help="Total target rows (overrides n-patients)"),
    positive_rate: float = typer.Option(0.07, "--positive-rate", help="Target positive diagnosis rate"),
    rows_per_chunk: int = typer.Option(100_000, "--rows-per-chunk", help="Rows per chunk for memory efficiency"),
    out: str = typer.Option("data/raw", "--out", help="Output directory"),
    seed: int = typer.Option(0, "--seed", help="Random seed for reproducibility"),
) -> None:
    """
    Generate synthetic patient-year data with realistic features.
    
    Creates a dataset with demographic, clinical, and temporal features.
    Data is saved to Parquet files partitioned by year for efficient querying.
    
    Example:
        python cli.py data-gen --n-patients 5000 --years 2018 2019 2020 --out data/raw
    """
    # Calculate n_patients from rows if specified
    if rows is not None:
        n_patients = rows // len(years)
        logger.info(f"Calculated n_patients={n_patients} from rows={rows} and years={len(years)}")
    
    config = DataGenConfig(
        n_patients=n_patients,
        years=years,
        positive_rate=positive_rate,
        rows_per_chunk=rows_per_chunk,
        seed=seed,
        output_dir=out,
    )
    
    generator = SyntheticDataGenerator(config)
    generator.generate()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    typer.run(generate)
