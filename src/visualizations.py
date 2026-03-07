# charts and visualizations for churn analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple

sns.set_style("whitegrid")


class ChurnVisualizer:
    """creates charts for the churn analysis report"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
    
    def plot_churn_by_contract_type(self, contract_col: str = 'ContractType', 
                                    churn_col: str = 'Churn', save_path: Optional[str] = None):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        df_plot = self.df.copy()
        if df_plot[churn_col].dtype == 'object':
            df_plot['churned'] = (df_plot[churn_col] == 'Yes').astype(int)
        else:
            df_plot['churned'] = df_plot[churn_col]
        
        churn_rate = df_plot.groupby(contract_col)['churned'].mean() * 100
        churn_rate.plot(kind='bar', ax=axes[0], color='steelblue')
        axes[0].set_title('Churn Rate by Contract Type')
        axes[0].set_ylabel('Churn Rate (%)')
        axes[0].tick_params(axis='x', rotation=45)
        
        churn_counts = df_plot.groupby([contract_col, 'churned']).size().unstack(fill_value=0)
        churn_counts.plot(kind='bar', stacked=True, ax=axes[1], color=['green', 'red'])
        axes[1].set_title('Customer Count by Contract Type')
        axes[1].legend(['Not Churned', 'Churned'])
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
    
    def plot_churn_by_tenure(self, tenure_col: str = 'Tenure', churn_col: str = 'Churn',
                            save_path: Optional[str] = None):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        df_plot = self.df.copy()
        if df_plot[churn_col].dtype == 'object':
            df_plot['churned'] = (df_plot[churn_col] == 'Yes').astype(int)
        else:
            df_plot['churned'] = df_plot[churn_col]
        
        df_plot['tenure_segment'] = pd.cut(
            df_plot[tenure_col], bins=[0, 6, 12, 24, 48, 100],
            labels=['0-6 mo', '6-12 mo', '1-2 yr', '2-4 yr', '4+ yr']
        )
        
        churned = df_plot[df_plot['churned'] == 1][tenure_col]
        not_churned = df_plot[df_plot['churned'] == 0][tenure_col]
        
        axes[0].hist(not_churned, bins=20, alpha=0.7, label='Not Churned', color='green')
        axes[0].hist(churned, bins=20, alpha=0.7, label='Churned', color='red')
        axes[0].set_title('Tenure Distribution')
        axes[0].legend()
        
        churn_by_segment = df_plot.groupby('tenure_segment')['churned'].mean() * 100
        churn_by_segment.plot(kind='bar', ax=axes[1], color='steelblue')
        axes[1].set_title('Churn Rate by Tenure')
        axes[1].set_ylabel('Churn Rate (%)')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
    
    def plot_churn_probability_distribution(self, prob_col: str = 'churn_probability',
                                           save_path: Optional[str] = None):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].hist(self.df[prob_col], bins=30, color='steelblue', edgecolor='white')
        axes[0].axvline(x=0.40, color='orange', linestyle='--', label='Medium Risk')
        axes[0].axvline(x=0.75, color='red', linestyle='--', label='High Risk')
        axes[0].set_title('Churn Probability Distribution')
        axes[0].legend()
        
        if 'risk_category' in self.df.columns:
            risk_counts = self.df['risk_category'].value_counts()
            colors = ['green', 'orange', 'red']
            axes[1].pie(risk_counts.values, labels=risk_counts.index, colors=colors, autopct='%1.1f%%')
            axes[1].set_title('Risk Segmentation')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
    
    def plot_feature_importance(self, feature_importance: pd.DataFrame, top_n: int = 10,
                               save_path: Optional[str] = None):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        top_features = feature_importance.head(top_n).sort_values('importance')
        ax.barh(top_features['feature'], top_features['importance'], color='steelblue')
        ax.set_xlabel('Importance')
        ax.set_title(f'Top {top_n} Churn Drivers')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
    
    def create_full_report(self, feature_importance: pd.DataFrame, confusion_matrix: np.ndarray,
                          metrics: dict, save_dir: str = 'reports/'):
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        self.plot_churn_by_contract_type(save_path=f'{save_dir}churn_by_contract.png')
        self.plot_churn_by_tenure(save_path=f'{save_dir}churn_by_tenure.png')
        
        if 'churn_probability' in self.df.columns:
            self.plot_churn_probability_distribution(save_path=f'{save_dir}probability_dist.png')
        
        self.plot_feature_importance(feature_importance, save_path=f'{save_dir}feature_importance.png')
        
        plt.close('all')
        print(f"Report saved to {save_dir}")