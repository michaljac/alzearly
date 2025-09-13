"""
Data generation package for synthetic patient-year datasets.

Implements R-DATA-001 v1.0: Synthetic patient-year tabular data with ≥20 mixed features,
configurable prevalence (5-10%), scalable via patients/years parameters,
stored in columnar/partitioned Parquet format.
"""

# Avoid circular imports by not importing generate here
__all__ = []
