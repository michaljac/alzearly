#!/usr/bin/env python3
"""
Slice Analysis Implementation for Alzheimer's Prediction Model

This script implements the slice analysis described in the training error analysis report,
providing practical tools to analyze model performance across demographic subgroups.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

class SliceAnalyzer:
    """Class for performing slice analysis across demographic subgroups."""
    
    def __init__(self, df, y_true, y_pred, sensitive_features=None):
        """
        Initialize the slice analyzer.
        
        Args:
            df: DataFrame with demographic and clinical features
            y_true: True labels
            y_pred: Model predictions (probabilities)
            sensitive_features: Dictionary of sensitive feature definitions
        """
        self.df = df.copy()
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.sensitive_features = sensitive_features or self._default_sensitive_features()
        self.results = {}
        
    def _default_sensitive_features(self):
        """Define default sensitive features for Alzheimer's analysis."""
        return {
            'age_groups': {
                'Young (50-65)': lambda x: (x['age'] >= 50) & (x['age'] <= 65),
                'Middle (66-80)': lambda x: (x['age'] >= 66) & (x['age'] <= 80),
                'Elderly (81+)': lambda x: x['age'] >= 81
            },
            'gender': {
                'Male': lambda x: x['gender'] == 'M',
                'Female': lambda x: x['gender'] == 'F'
            },
            'education': {
                'Low (<12 years)': lambda x: x['education_years'] < 12,
                'Medium (12-16 years)': lambda x: (x['education_years'] >= 12) & (x['education_years'] <= 16),
                'High (>16 years)': lambda x: x['education_years'] > 16
            }
        }
    
    def analyze_slice(self, slice_name, mask):
        """Analyze performance for a specific slice."""
        if mask.sum() == 0:
            return None
            
        y_true_slice = self.y_true[mask]
        y_pred_slice = self.y_pred[mask]
        
        # Calculate metrics
        auc = roc_auc_score(y_true_slice, y_pred_slice)
        ap = average_precision_score(y_true_slice, y_pred_slice)
        positive_rate = y_true_slice.mean()
        
        # Calculate additional metrics
        precision, recall, _ = precision_recall_curve(y_true_slice, y_pred_slice)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        max_f1 = np.max(f1_scores)
        
        return {
            'sample_size': mask.sum(),
            'auc': auc,
            'average_precision': ap,
            'positive_rate': positive_rate,
            'max_f1': max_f1,
            'prediction_mean': y_pred_slice.mean(),
            'prediction_std': y_pred_slice.std()
        }
    
    def analyze_all_slices(self):
        """Analyze performance across all defined slices."""
        all_results = {}
        
        for feature_type, slices in self.sensitive_features.items():
            feature_results = {}
            
            for slice_name, mask_func in slices.items():
                mask = mask_func(self.df)
                result = self.analyze_slice(slice_name, mask)
                
                if result is not None:
                    feature_results[slice_name] = result
            
            all_results[feature_type] = feature_results
        
        self.results = all_results
        return all_results
    
    def calculate_fairness_metrics(self):
        """Calculate fairness metrics across slices."""
        fairness_metrics = {}
        
        for feature_type, slices in self.results.items():
            if len(slices) < 2:
                continue
                
            # Calculate demographic parity difference
            positive_rates = [slice_data['positive_rate'] for slice_data in slices.values()]
            demographic_parity_diff = max(positive_rates) - min(positive_rates)
            
            # Calculate equalized odds difference (simplified)
            aucs = [slice_data['auc'] for slice_data in slices.values()]
            equalized_odds_diff = max(aucs) - min(aucs)
            
            fairness_metrics[feature_type] = {
                'demographic_parity_difference': demographic_parity_diff,
                'equalized_odds_difference': equalized_odds_diff,
                'auc_gap': equalized_odds_diff,
                'positive_rate_gap': demographic_parity_diff
            }
        
        return fairness_metrics
    
    def generate_report(self):
        """Generate a comprehensive slice analysis report."""
        if not self.results:
            self.analyze_all_slices()
        
        fairness_metrics = self.calculate_fairness_metrics()
        
        print("=" * 80)
        print("SLICE ANALYSIS REPORT - Alzheimer's Prediction Model")
        print("=" * 80)
        
        # Overall performance
        overall_auc = roc_auc_score(self.y_true, self.y_pred)
        overall_ap = average_precision_score(self.y_true, self.y_pred)
        overall_positive_rate = self.y_true.mean()
        
        print(f"\n📊 OVERALL PERFORMANCE")
        print(f"   AUC: {overall_auc:.3f}")
        print(f"   Average Precision: {overall_ap:.3f}")
        print(f"   Positive Rate: {overall_positive_rate:.3f}")
        print(f"   Sample Size: {len(self.y_true):,}")
        
        # Slice performance
        print(f"\n🔍 SLICE PERFORMANCE ANALYSIS")
        for feature_type, slices in self.results.items():
            print(f"\n   {feature_type.upper().replace('_', ' ')}:")
            print(f"   {'Slice':<20} {'Size':<8} {'AUC':<8} {'AP':<8} {'Pos Rate':<10} {'Gap':<8}")
            print(f"   {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*8}")
            
            baseline_auc = None
            for slice_name, metrics in slices.items():
                if baseline_auc is None:
                    baseline_auc = metrics['auc']
                    gap = 0.0
                else:
                    gap = metrics['auc'] - baseline_auc
                
                print(f"   {slice_name:<20} {metrics['sample_size']:<8} "
                      f"{metrics['auc']:<8.3f} {metrics['average_precision']:<8.3f} "
                      f"{metrics['positive_rate']:<10.3f} {gap:<8.3f}")
        
        # Fairness analysis
        print(f"\n⚖️  FAIRNESS ANALYSIS")
        for feature_type, metrics in fairness_metrics.items():
            print(f"\n   {feature_type.upper().replace('_', ' ')}:")
            print(f"   Demographic Parity Difference: {metrics['demographic_parity_difference']:.3f}")
            print(f"   Equalized Odds Difference: {metrics['equalized_odds_difference']:.3f}")
            print(f"   AUC Gap: {metrics['auc_gap']:.3f}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS")
        self._generate_recommendations(fairness_metrics)
        
        return {
            'overall_performance': {
                'auc': overall_auc,
                'average_precision': overall_ap,
                'positive_rate': overall_positive_rate
            },
            'slice_results': self.results,
            'fairness_metrics': fairness_metrics
        }
    
    def _generate_recommendations(self, fairness_metrics):
        """Generate specific recommendations based on fairness metrics."""
        recommendations = []
        
        for feature_type, metrics in fairness_metrics.items():
            if metrics['auc_gap'] > 0.1:
                recommendations.append(f"   ⚠️  Large AUC gap ({metrics['auc_gap']:.3f}) in {feature_type}")
            
            if metrics['demographic_parity_difference'] > 0.15:
                recommendations.append(f"   ⚠️  High demographic parity difference ({metrics['demographic_parity_difference']:.3f}) in {feature_type}")
        
        if not recommendations:
            print("   ✅ No significant fairness issues detected")
        else:
            for rec in recommendations:
                print(rec)
            
            print("\n   🔧 Suggested Actions:")
            print("   - Implement subgroup-aware training")
            print("   - Add fairness constraints to model training")
            print("   - Increase representation of underrepresented groups")
            print("   - Create subgroup-specific features")
    
    def plot_slice_performance(self, save_path=None):
        """Create visualization of slice performance."""
        if not self.results:
            self.analyze_all_slices()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Slice Analysis: Model Performance Across Subgroups', fontsize=16)
        
        # Plot 1: AUC by slice
        ax1 = axes[0, 0]
        slice_data = []
        for feature_type, slices in self.results.items():
            for slice_name, metrics in slices.items():
                slice_data.append({
                    'feature_type': feature_type,
                    'slice_name': slice_name,
                    'auc': metrics['auc'],
                    'sample_size': metrics['sample_size']
                })
        
        slice_df = pd.DataFrame(slice_data)
        
        for feature_type in slice_df['feature_type'].unique():
            data = slice_df[slice_df['feature_type'] == feature_type]
            ax1.bar(data['slice_name'], data['auc'], alpha=0.7, label=feature_type)
        
        ax1.set_title('AUC by Subgroup')
        ax1.set_ylabel('AUC')
        ax1.tick_params(axis='x', rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Sample sizes
        ax2 = axes[0, 1]
        for feature_type in slice_df['feature_type'].unique():
            data = slice_df[slice_df['feature_type'] == feature_type]
            ax2.bar(data['slice_name'], data['sample_size'], alpha=0.7, label=feature_type)
        
        ax2.set_title('Sample Size by Subgroup')
        ax2.set_ylabel('Sample Size')
        ax2.tick_params(axis='x', rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Fairness gaps
        ax3 = axes[1, 0]
        fairness_data = []
        for feature_type, slices in self.results.items():
            if len(slices) >= 2:
                aucs = [metrics['auc'] for metrics in slices.values()]
                fairness_data.append({
                    'feature_type': feature_type,
                    'auc_gap': max(aucs) - min(aucs)
                })
        
        if fairness_data:
            fairness_df = pd.DataFrame(fairness_data)
            ax3.bar(fairness_df['feature_type'], fairness_df['auc_gap'])
            ax3.set_title('AUC Gap by Feature Type')
            ax3.set_ylabel('AUC Gap')
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Positive rates
        ax4 = axes[1, 1]
        for feature_type, slices in self.results.items():
            slice_names = list(slices.keys())
            positive_rates = [metrics['positive_rate'] for metrics in slices.values()]
            ax4.bar(slice_names, positive_rates, alpha=0.7, label=feature_type)
        
        ax4.set_title('Positive Rate by Subgroup')
        ax4.set_ylabel('Positive Rate')
        ax4.tick_params(axis='x', rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Plot saved to: {save_path}")
        
        plt.show()

def run_slice_analysis_example():
    """Run an example slice analysis with synthetic data."""
    print("🧠 Running Slice Analysis Example...")
    
    # Generate synthetic data for demonstration
    np.random.seed(42)
    n_samples = 5000
    
    # Create synthetic demographic data
    data = {
        'age': np.random.normal(70, 15, n_samples).clip(50, 95),
        'gender': np.random.choice(['M', 'F'], n_samples, p=[0.45, 0.55]),
        'education_years': np.random.normal(14, 4, n_samples).clip(6, 22),
        'cognitive_score': np.random.normal(25, 8, n_samples).clip(10, 40),
        'biomarker_level': np.random.normal(1000, 300, n_samples).clip(400, 2000)
    }
    
    df = pd.DataFrame(data)
    
    # Create synthetic labels with demographic biases
    # Simulate higher Alzheimer's risk for older, less educated, and female patients
    age_risk = (df['age'] - 50) / 45  # Normalized age risk
    education_risk = (22 - df['education_years']) / 16  # Lower education = higher risk
    gender_risk = (df['gender'] == 'F').astype(float) * 0.1  # Slightly higher risk for females
    
    # Combine risk factors
    risk_score = 0.4 * age_risk + 0.3 * education_risk + 0.1 * gender_risk + 0.2 * np.random.random(n_samples)
    y_true = (risk_score > np.percentile(risk_score, 85)).astype(int)  # 15% positive rate
    
    # Create synthetic predictions with some bias
    y_pred = risk_score + np.random.normal(0, 0.1, n_samples)
    y_pred = 1 / (1 + np.exp(-y_pred))  # Convert to probabilities
    
    # Run slice analysis
    analyzer = SliceAnalyzer(df, y_true, y_pred)
    results = analyzer.generate_report()
    
    # Create visualization
    analyzer.plot_slice_performance('slice_analysis_plot.png')
    
    return results

if __name__ == "__main__":
    # Run the example analysis
    results = run_slice_analysis_example()
    
    print("\n" + "=" * 80)
    print("✅ Slice Analysis Complete!")
    print("=" * 80)
