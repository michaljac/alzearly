"""
Clinical Unit Conversion Module

This module provides utilities for handling clinical measurements with proper
unit conversions and validation. It supports common clinical units and their
conversions (e.g., mg/dL vs. mmol/L, cm vs. inches).
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging
from pint import UnitRegistry, Quantity
import warnings

logger = logging.getLogger(__name__)

# Initialize Pint unit registry
ureg = UnitRegistry()

@dataclass
class ClinicalField:
    """Represents a clinical field with its units and conversion rules."""
    name: str
    standard_unit: str
    allowed_units: List[str]
    conversion_factors: Dict[str, float]
    description: str = ""
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    
    def convert_to_standard(self, value: float, from_unit: str) -> float:
        """Convert value from given unit to standard unit."""
        if from_unit == self.standard_unit:
            return value
        
        if from_unit not in self.conversion_factors:
            raise ValueError(f"Unknown unit '{from_unit}' for field '{self.name}'. "
                           f"Allowed units: {self.allowed_units}")
        
        return value * self.conversion_factors[from_unit]
    
    def convert_from_standard(self, value: float, to_unit: str) -> float:
        """Convert value from standard unit to given unit."""
        if to_unit == self.standard_unit:
            return value
        
        if to_unit not in self.conversion_factors:
            raise ValueError(f"Unknown unit '{to_unit}' for field '{self.name}'. "
                           f"Allowed units: {self.allowed_units}")
        
        return value / self.conversion_factors[to_unit]
    
    def validate_value(self, value: float, unit: str) -> bool:
        """Validate if value is within acceptable range for the unit."""
        if self.min_value is None and self.max_value is None:
            return True
        
        # Convert to standard unit for validation
        std_value = self.convert_to_standard(value, unit)
        
        if self.min_value is not None and std_value < self.min_value:
            return False
        if self.max_value is not None and std_value > self.max_value:
            return False
        
        return True

class ClinicalUnitConverter:
    """Handles clinical unit conversions and validation."""
    
    def __init__(self):
        self.fields = self._initialize_clinical_fields()
    
    def _initialize_clinical_fields(self) -> Dict[str, ClinicalField]:
        """Initialize standard clinical fields with their units and conversions."""
        return {
            # Blood glucose
            'glucose': ClinicalField(
                name='glucose',
                standard_unit='mg/dL',
                allowed_units=['mg/dL', 'mmol/L'],
                conversion_factors={'mmol/L': 18.018},  # 1 mmol/L = 18.018 mg/dL
                description='Blood glucose level',
                min_value=20,  # mg/dL
                max_value=1000  # mg/dL
            ),
            
            # Cholesterol
            'total_cholesterol': ClinicalField(
                name='total_cholesterol',
                standard_unit='mg/dL',
                allowed_units=['mg/dL', 'mmol/L'],
                conversion_factors={'mmol/L': 38.67},  # 1 mmol/L = 38.67 mg/dL
                description='Total cholesterol level',
                min_value=50,  # mg/dL
                max_value=1000  # mg/dL
            ),
            
            'hdl_cholesterol': ClinicalField(
                name='hdl_cholesterol',
                standard_unit='mg/dL',
                allowed_units=['mg/dL', 'mmol/L'],
                conversion_factors={'mmol/L': 38.67},
                description='HDL cholesterol level',
                min_value=10,  # mg/dL
                max_value=200  # mg/dL
            ),
            
            'ldl_cholesterol': ClinicalField(
                name='ldl_cholesterol',
                standard_unit='mg/dL',
                allowed_units=['mg/dL', 'mmol/L'],
                conversion_factors={'mmol/L': 38.67},
                description='LDL cholesterol level',
                min_value=10,  # mg/dL
                max_value=500  # mg/dL
            ),
            
            # Blood pressure
            'systolic_bp': ClinicalField(
                name='systolic_bp',
                standard_unit='mmHg',
                allowed_units=['mmHg', 'kPa'],
                conversion_factors={'kPa': 7.5},  # 1 kPa = 7.5 mmHg
                description='Systolic blood pressure',
                min_value=50,  # mmHg
                max_value=300  # mmHg
            ),
            
            'diastolic_bp': ClinicalField(
                name='diastolic_bp',
                standard_unit='mmHg',
                allowed_units=['mmHg', 'kPa'],
                conversion_factors={'kPa': 7.5},
                description='Diastolic blood pressure',
                min_value=30,  # mmHg
                max_value=200  # mmHg
            ),
            
            # Height and weight
            'height': ClinicalField(
                name='height',
                standard_unit='cm',
                allowed_units=['cm', 'inches', 'm'],
                conversion_factors={'inches': 2.54, 'm': 100},  # 1 inch = 2.54 cm, 1 m = 100 cm
                description='Height',
                min_value=50,  # cm
                max_value=250  # cm
            ),
            
            'weight': ClinicalField(
                name='weight',
                standard_unit='kg',
                allowed_units=['kg', 'lbs'],
                conversion_factors={'lbs': 0.453592},  # 1 lb = 0.453592 kg
                description='Weight',
                min_value=10,  # kg
                max_value=300  # kg
            ),
            
            # BMI
            'bmi': ClinicalField(
                name='bmi',
                standard_unit='kg/m²',
                allowed_units=['kg/m²'],
                conversion_factors={},
                description='Body Mass Index',
                min_value=10,  # kg/m²
                max_value=80  # kg/m²
            ),
            
            # Age
            'age': ClinicalField(
                name='age',
                standard_unit='years',
                allowed_units=['years', 'months'],
                conversion_factors={'months': 1/12},  # 1 month = 1/12 years
                description='Age',
                min_value=0,  # years
                max_value=150  # years
            )
        }
    
    def get_field(self, field_name: str) -> ClinicalField:
        """Get clinical field by name."""
        if field_name not in self.fields:
            raise ValueError(f"Unknown clinical field: {field_name}. "
                           f"Available fields: {list(self.fields.keys())}")
        return self.fields[field_name]
    
    def convert_value(self, value: float, field_name: str, from_unit: str, to_unit: str) -> float:
        """Convert a value from one unit to another for a specific field."""
        field = self.get_field(field_name)
        
        # Convert to standard unit first, then to target unit
        std_value = field.convert_to_standard(value, from_unit)
        return field.convert_from_standard(std_value, to_unit)
    
    def standardize_dataframe(self, df: pd.DataFrame, unit_mapping: Dict[str, str]) -> pd.DataFrame:
        """
        Standardize all clinical values in a dataframe to their standard units.
        
        Args:
            df: DataFrame containing clinical data
            unit_mapping: Dict mapping column names to their units
                          e.g., {'glucose': 'mmol/L', 'height': 'inches'}
        
        Returns:
            DataFrame with all values converted to standard units
        """
        df_std = df.copy()
        
        for column, unit in unit_mapping.items():
            if column in df.columns and column in self.fields:
                field = self.fields[column]
                
                # Convert values to standard unit
                df_std[column] = df[column].apply(
                    lambda x: field.convert_to_standard(x, unit) if pd.notna(x) else x
                )
                
                # Converted unit
        
        return df_std
    
    def validate_dataframe(self, df: pd.DataFrame, unit_mapping: Dict[str, str]) -> Dict[str, List[int]]:
        """
        Validate clinical values in a dataframe.
        
        Returns:
            Dict mapping field names to lists of invalid row indices
        """
        invalid_rows = {}
        
        for column, unit in unit_mapping.items():
            if column in df.columns and column in self.fields:
                field = self.fields[column]
                invalid_indices = []
                
                for idx, value in enumerate(df[column]):
                    if pd.notna(value) and not field.validate_value(value, unit):
                        invalid_indices.append(idx)
                
                if invalid_indices:
                    invalid_rows[column] = invalid_indices
                    logger.warning(f"Found {len(invalid_indices)} invalid values for {column}")
        
        return invalid_rows
    
    def get_unit_info(self, field_name: str) -> Dict[str, any]:
        """Get information about a clinical field's units."""
        field = self.get_field(field_name)
        return {
            'name': field.name,
            'standard_unit': field.standard_unit,
            'allowed_units': field.allowed_units,
            'description': field.description,
            'min_value': field.min_value,
            'max_value': field.max_value
        }

# Global converter instance
clinical_converter = ClinicalUnitConverter()

def convert_clinical_value(value: float, field_name: str, from_unit: str, to_unit: str) -> float:
    """Convenience function for converting a single clinical value."""
    return clinical_converter.convert_value(value, field_name, from_unit, to_unit)

def standardize_clinical_data(df: pd.DataFrame, unit_mapping: Dict[str, str]) -> pd.DataFrame:
    """Convenience function for standardizing clinical data."""
    return clinical_converter.standardize_dataframe(df, unit_mapping)

def validate_clinical_data(df: pd.DataFrame, unit_mapping: Dict[str, str]) -> Dict[str, List[int]]:
    """Convenience function for validating clinical data."""
    return clinical_converter.validate_dataframe(df, unit_mapping)
