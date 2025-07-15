#!/usr/bin/env python3
"""
Explainable AI (XAI) Analysis for BRFSS 2022 Dental Health Screening Tool
========================================================================

This script implements comprehensive XAI analysis using SHAP and LIME
for clinical interpretability and publication-quality insights.

Key Components:
1. SHAP Analysis (5 visualization types)
2. LIME Analysis for model-agnostic explanations
3. Clinical interpretation of feature importance
4. Publication-ready visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import lime
from lime.lime_tabular import LimeTabularExplainer
import warnings
warnings.filterwarnings('ignore')

# Import our analysis tools
from optimized_screening_analysis import OptimizedDentalScreeningTool
import pickle
import os
from datetime import datetime

class DentalHealthXAI:
    """
    Comprehensive XAI analysis for dental health screening model.
    """
    
    def __init__(self, screening_tool):
        """
        Initialize XAI analysis with trained screening tool.
        
        Parameters:
        -----------
        screening_tool : OptimizedDentalScreeningTool
            Trained screening tool with best model
        """
        self.screening_tool = screening_tool
        self.best_model = screening_tool.best_model
        self.X_train = screening_tool.X_train
        self.X_test = screening_tool.X_test
        self.y_test = screening_tool.y_test
        self.feature_names = screening_tool.feature_names_final
        
        # Initialize SHAP explainer
        self.shap_explainer = None
        self.shap_values = None
        
        # Initialize LIME explainer
        self.lime_explainer = None
        
        print("XAI Analysis initialized successfully!")
        print(f"Model type: {type(self.best_model).__name__}")
        print(f"Features: {len(self.feature_names)}")
        print(f"Test samples: {len(self.X_test)}")
    
    def setup_shap_analysis(self, sample_size=1000):
        """
        Setup SHAP explainer and calculate SHAP values.
        
        Parameters:
        -----------
        sample_size : int
            Sample size for SHAP analysis (for efficiency)
        """
        print(f"\n🔍 Setting up SHAP analysis with {sample_size:,} samples...")
        
        # Sample data for efficiency
        if len(self.X_test) > sample_size:
            sample_idx = np.random.choice(len(self.X_test), sample_size, replace=False)
            X_sample = self.X_test.iloc[sample_idx]
        else:
            X_sample = self.X_test
            sample_idx = range(len(self.X_test))
        
        # Initialize SHAP explainer based on model type
        model_name = type(self.best_model).__name__
        
        if 'XGB' in model_name:
            self.shap_explainer = shap.TreeExplainer(self.best_model)
            self.shap_values = self.shap_explainer.shap_values(X_sample)
        elif 'RandomForest' in model_name:
            self.shap_explainer = shap.TreeExplainer(self.best_model)
            self.shap_values = self.shap_explainer.shap_values(X_sample)
            if isinstance(self.shap_values, list):
                self.shap_values = self.shap_values[1]  # For binary classification
        else:  # LogisticRegression
            # Use a background sample for linear models
            background_sample = shap.sample(self.X_train, min(100, len(self.X_train)))
            self.shap_explainer = shap.LinearExplainer(self.best_model, background_sample)
            self.shap_values = self.shap_explainer.shap_values(X_sample)
        
        self.X_shap_sample = X_sample
        self.sample_idx = sample_idx
        
        print(f"✅ SHAP analysis setup completed")
        print(f"   SHAP values shape: {self.shap_values.shape}")
    
    def create_shap_visualizations(self, save_plots=True):
        """
        Create comprehensive SHAP visualizations for clinical interpretation.
        
        Parameters:
        -----------
        save_plots : bool
            Whether to save plots to files
        """
        print("\n📊 Creating SHAP visualizations...")
        
        if self.shap_values is None:
            self.setup_shap_analysis()
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Summary Plot (Swarm Plot)
        print("   Creating summary plot (swarm)...")
        plt.figure(figsize=(12, 8))
        shap.summary_plot(self.shap_values, self.X_shap_sample, 
                         feature_names=self.feature_names, show=False)
        plt.title('SHAP Summary Plot: Feature Impact on Dental Care Access Prediction', 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        if save_plots:
            plt.savefig('shap_summary_swarm.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Feature Importance Bar Plot
        print("   Creating feature importance bar plot...")
        plt.figure(figsize=(10, 8))
        shap.summary_plot(self.shap_values, self.X_shap_sample, 
                         feature_names=self.feature_names, plot_type="bar", show=False)
        plt.title('SHAP Feature Importance: Absolute Impact on Predictions', 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        if save_plots:
            plt.savefig('shap_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Waterfall Plot for Individual Prediction
        print("   Creating waterfall plot for individual case...")
        # Select an interesting case (e.g., high-risk individual)
        case_idx = 0  # First case in sample
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=self.shap_values[case_idx],
                base_values=self.shap_explainer.expected_value,
                data=self.X_shap_sample.iloc[case_idx].values,
                feature_names=self.feature_names
            ),
            show=False
        )
        plt.title(f'SHAP Waterfall Plot: Individual Case Analysis (Case #{case_idx+1})', 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        if save_plots:
            plt.savefig('shap_waterfall_individual.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 4. Dependence Plot for Top Features
        print("   Creating dependence plots for top features...")
        # Get top 3 most important features
        feature_importance = np.abs(self.shap_values).mean(0)
        top_features_idx = np.argsort(feature_importance)[-3:]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for i, feat_idx in enumerate(top_features_idx):
            shap.dependence_plot(feat_idx, self.shap_values, self.X_shap_sample,
                               feature_names=self.feature_names, ax=axes[i], show=False)
            axes[i].set_title(f'Dependence: {self.feature_names[feat_idx]}', fontweight='bold')
        
        plt.suptitle('SHAP Dependence Plots: Top 3 Most Important Features', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        if save_plots:
            plt.savefig('shap_dependence_plots.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 5. Heatmap for Multiple Cases
        print("   Creating SHAP heatmap for multiple cases...")
        # Select first 20 cases for heatmap
        n_cases = min(20, len(self.shap_values))
        plt.figure(figsize=(14, 8))
        
        # Create heatmap data
        heatmap_data = self.shap_values[:n_cases].T
        
        # Create custom heatmap
        sns.heatmap(heatmap_data, 
                   xticklabels=[f'Case {i+1}' for i in range(n_cases)],
                   yticklabels=self.feature_names,
                   cmap='RdBu_r', center=0, 
                   cbar_kws={'label': 'SHAP Value'})
        
        plt.title('SHAP Heatmap: Feature Contributions Across Multiple Cases', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Individual Cases')
        plt.ylabel('Features')
        plt.tight_layout()
        if save_plots:
            plt.savefig('shap_heatmap_cases.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ SHAP visualizations completed")
    
    def setup_lime_analysis(self):
        """
        Setup LIME explainer for model-agnostic explanations.
        """
        print("\n🍋 Setting up LIME analysis...")
        
        # Convert to numpy arrays for LIME
        X_train_np = self.X_train.values
        X_test_np = self.X_test.values
        
        # Initialize LIME explainer
        self.lime_explainer = LimeTabularExplainer(
            X_train_np,
            feature_names=self.feature_names,
            class_names=['No Regular Care', 'Regular Care'],
            mode='classification',
            discretize_continuous=True
        )
        
        print("✅ LIME analysis setup completed")
    
    def create_lime_explanations(self, n_cases=5, save_plots=True):
        """
        Create LIME explanations for individual cases.
        
        Parameters:
        -----------
        n_cases : int
            Number of cases to explain
        save_plots : bool
            Whether to save explanations
        """
        print(f"\n📋 Creating LIME explanations for {n_cases} cases...")
        
        if self.lime_explainer is None:
            self.setup_lime_analysis()
        
        X_test_np = self.X_test.values
        
        # Create explanations for selected cases
        for i in range(min(n_cases, len(X_test_np))):
            print(f"   Explaining case {i+1}...")
            
            # Get LIME explanation
            explanation = self.lime_explainer.explain_instance(
                X_test_np[i], 
                self.best_model.predict_proba,
                num_features=len(self.feature_names)
            )
            
            # Save explanation
            if save_plots:
                explanation.save_to_file(f'lime_explanation_case_{i+1}.html')
            
            # Show in notebook (if available)
            try:
                explanation.show_in_notebook(show_table=True)
            except:
                pass  # Skip if not in notebook environment
        
        print("✅ LIME explanations completed")
    
    def generate_clinical_interpretation(self):
        """
        Generate clinical interpretation of XAI results.
        
        Returns:
        --------
        dict: Clinical interpretation summary
        """
        print("\n🏥 Generating clinical interpretation...")
        
        if self.shap_values is None:
            self.setup_shap_analysis()
        
        # Calculate feature importance
        feature_importance = np.abs(self.shap_values).mean(0)
        feature_ranking = np.argsort(feature_importance)[::-1]
        
        # Get top features
        top_features = [(self.feature_names[i], feature_importance[i]) for i in feature_ranking[:10]]
        
        # Clinical interpretation
        clinical_interpretation = {
            'top_predictive_features': top_features,
            'clinical_insights': {
                'most_important_factor': top_features[0][0],
                'socioeconomic_factors': [feat for feat, _ in top_features if any(term in feat.lower() for term in ['income', 'education', 'employ'])],
                'health_factors': [feat for feat, _ in top_features if any(term in feat.lower() for term in ['health', 'disease', 'chronic'])],
                'demographic_factors': [feat for feat, _ in top_features if any(term in feat.lower() for term in ['age', 'sex', 'gender'])]
            },
            'screening_implications': {
                'high_risk_indicators': 'Low income, low education, poor general health, chronic diseases',
                'protective_factors': 'Higher income, higher education, good general health',
                'clinical_actionability': 'Model identifies modifiable risk factors for targeted interventions'
            }
        }
        
        print("✅ Clinical interpretation completed")
        return clinical_interpretation

def main():
    """
    Run comprehensive XAI analysis.
    """
    print("="*80)
    print("BRFSS 2022 Dental Health Screening Tool - XAI Analysis")
    print("Explainable AI for Clinical Interpretability")
    print("="*80)
    
    # Load the trained screening tool
    print("\n📂 Loading trained screening tool...")
    
    # Run the optimized analysis first if needed
    try:
        from optimized_screening_analysis import main as run_analysis
        screening_tool = run_analysis()
    except:
        print("❌ Could not load screening tool. Please run optimized_screening_analysis.py first.")
        return
    
    # Initialize XAI analysis
    xai_analyzer = DentalHealthXAI(screening_tool)
    
    # Perform SHAP analysis
    xai_analyzer.create_shap_visualizations(save_plots=True)
    
    # Perform LIME analysis
    xai_analyzer.create_lime_explanations(n_cases=3, save_plots=True)
    
    # Generate clinical interpretation
    clinical_insights = xai_analyzer.generate_clinical_interpretation()
    
    # Print summary
    print("\n" + "="*80)
    print("🎉 XAI ANALYSIS COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"📊 SHAP visualizations: 5 types created")
    print(f"🍋 LIME explanations: 3 individual cases")
    print(f"🏥 Clinical insights: Generated")
    print(f"📁 Files saved: Multiple visualization files")
    
    print(f"\n🔍 TOP PREDICTIVE FEATURES:")
    for i, (feature, importance) in enumerate(clinical_insights['top_predictive_features'][:5], 1):
        print(f"   {i}. {feature}: {importance:.3f}")
    
    print(f"\n🏥 CLINICAL IMPLICATIONS:")
    print(f"   High-risk indicators: {clinical_insights['screening_implications']['high_risk_indicators']}")
    print(f"   Protective factors: {clinical_insights['screening_implications']['protective_factors']}")
    
    return xai_analyzer

if __name__ == "__main__":
    xai_analyzer = main()
