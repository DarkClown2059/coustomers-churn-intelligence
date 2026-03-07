# business analysis helpers

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class BusinessMetrics:
    """container for key business metrics"""
    total_customers: int
    churn_rate: float
    high_risk_customers: int
    medium_risk_customers: int
    low_risk_customers: int
    revenue_at_risk: float
    average_monthly_charges: float
    average_tenure: float


class ChurnAnalyzer:
    """calculates business metrics and generates insights from churn data"""
    
    def __init__(self, df: pd.DataFrame, churn_probabilities: Optional[np.ndarray] = None,
                 risk_categories: Optional[List[str]] = None):
        self.df = df.copy()
        if churn_probabilities is not None:
            self.df['churn_probability'] = churn_probabilities
        if risk_categories is not None:
            self.df['risk_category'] = risk_categories
    
    def calculate_business_metrics(self, monthly_charges_col: str = 'MonthlyCharges',
                                   tenure_col: str = 'Tenure', churn_col: str = 'Churn') -> BusinessMetrics:
        total_customers = len(self.df)
        
        if churn_col in self.df.columns:
            if self.df[churn_col].dtype == 'object':
                churned = (self.df[churn_col] == 'Yes').sum()
            else:
                churned = self.df[churn_col].sum()
            churn_rate = churned / total_customers
        elif 'churn_probability' in self.df.columns:
            churn_rate = self.df['churn_probability'].mean()
        else:
            churn_rate = 0.0
        
        if 'risk_category' in self.df.columns:
            high_risk = (self.df['risk_category'] == 'High Risk').sum()
            medium_risk = (self.df['risk_category'] == 'Medium Risk').sum()
            low_risk = (self.df['risk_category'] == 'Low Risk').sum()
        else:
            high_risk = medium_risk = low_risk = 0
        
        if 'risk_category' in self.df.columns and monthly_charges_col in self.df.columns:
            revenue_at_risk = self.df[self.df['risk_category'] == 'High Risk'][monthly_charges_col].sum()
        else:
            revenue_at_risk = 0.0
        
        avg_monthly = self.df[monthly_charges_col].mean() if monthly_charges_col in self.df.columns else 0.0
        avg_tenure = self.df[tenure_col].mean() if tenure_col in self.df.columns else 0.0
        
        return BusinessMetrics(
            total_customers=total_customers, churn_rate=churn_rate,
            high_risk_customers=high_risk, medium_risk_customers=medium_risk,
            low_risk_customers=low_risk, revenue_at_risk=revenue_at_risk,
            average_monthly_charges=avg_monthly, average_tenure=avg_tenure
        )
    
    def analyze_churn_by_feature(self, feature: str, churn_col: str = 'Churn') -> pd.DataFrame:
        df_analysis = self.df.copy()
        if churn_col in df_analysis.columns:
            if df_analysis[churn_col].dtype == 'object':
                df_analysis['churned'] = (df_analysis[churn_col] == 'Yes').astype(int)
            else:
                df_analysis['churned'] = df_analysis[churn_col]
        else:
            df_analysis['churned'] = (df_analysis['churn_probability'] > 0.5).astype(int)
        
        analysis = df_analysis.groupby(feature).agg({'churned': ['sum', 'count', 'mean']}).round(4)
        analysis.columns = ['churned_count', 'total_count', 'churn_rate']
        return analysis.reset_index()
    
    def get_high_risk_customers(self) -> pd.DataFrame:
        if 'risk_category' not in self.df.columns:
            return pd.DataFrame()
        return self.df[self.df['risk_category'] == 'High Risk']
    
    def generate_business_insights(self, feature_importance: Optional[pd.DataFrame] = None) -> str:
        metrics = self.calculate_business_metrics()
        
        insights = []
        insights.append("=" * 60)
        insights.append("CUSTOMER CHURN ANALYTICS - BUSINESS INSIGHTS")
        insights.append("=" * 60)
        
        insights.append(f"\n📊 KEY METRICS")
        insights.append(f"• Total Customers: {metrics.total_customers:,}")
        insights.append(f"• Churn Rate: {metrics.churn_rate:.1%}")
        insights.append(f"• Avg Monthly Charges: ${metrics.average_monthly_charges:.2f}")
        
        insights.append(f"\n⚠️ RISK SEGMENTATION")
        insights.append(f"• High Risk: {metrics.high_risk_customers:,} ({metrics.high_risk_customers/metrics.total_customers:.1%})")
        insights.append(f"• Medium Risk: {metrics.medium_risk_customers:,}")
        insights.append(f"• Low Risk: {metrics.low_risk_customers:,}")
        insights.append(f"• Monthly Revenue at Risk: ${metrics.revenue_at_risk:,.2f}")
        
        if feature_importance is not None and len(feature_importance) > 0:
            insights.append(f"\n🔍 TOP CHURN DRIVERS")
            for _, row in feature_importance.head(5).iterrows():
                insights.append(f"• {row['feature']}: {row['importance']:.4f}")
        
        insights.append(f"\n💡 RECOMMENDATIONS")
        insights.append("1. Focus retention efforts on high-risk customers")
        insights.append("2. Incentivize month-to-month customers to switch to annual contracts")
        insights.append("3. Implement proactive support outreach for at-risk segments")
        insights.append("4. Review pricing strategy for high-churn segments")
        
        return "\n".join(insights)