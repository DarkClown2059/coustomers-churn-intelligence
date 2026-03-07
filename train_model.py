#!/usr/bin/env python3
# trains the churn model
# note: had to fix data leakage bug - was scaling before train/test split

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import load_data, preprocess_data, save_preprocessing_artifacts
from src.model import ChurnModel
from src.analysis import ChurnAnalyzer
from src.visualizations import ChurnVisualizer


def main():
    print("=" * 60)
    print("CUSTOMER CHURN MODEL TRAINING")
    print("=" * 60)
    
    # create output folders
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    print("\nLoading data...")
    df = load_data("customer_churn_data.csv")
    df = preprocess_data(df)
    print(f"   Got {len(df):,} records")
    
    print("\nPreparing features...")
    y = df['Churn'].map({'Yes': 1, 'No': 0}).values
    
    df_features = df.drop(columns=['Churn', 'CustomerID'])
    categorical_cols = df_features.select_dtypes(include=['object']).columns.tolist()
    df_encoded = pd.get_dummies(df_features, columns=categorical_cols, drop_first=True)
    feature_names = df_encoded.columns.tolist()
    X_raw = df_encoded.values
    
    print(f"   Features: {len(feature_names)}")
    
    # important: split BEFORE scaling otherwise you get data leakage
    # (learned this the hard way - model was getting 100% accuracy lol)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Train: {len(X_train_raw)}, Test: {len(X_test_raw)}")
    
    print("   Scaling...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)  # only transform, don't fit!
    X = scaler.transform(X_raw)  # for later predictions
    
    save_preprocessing_artifacts(scaler, feature_names, 'models/')
    
    print("\nTraining model...")
    model = ChurnModel(model_type='random_forest')
    model.train(X_train, y_train, feature_names=feature_names)
    
    print("\nResults:")
    metrics = model.evaluate(X_test, y_test)
    for k, v in metrics.items():
        print(f"   {k}: {v:.4f}")
    
    print("\n" + model.get_classification_report(X_test, y_test))
    model.save('models/churn_model.pkl')
    
    print("\nGenerating predictions for all customers...")
    probs = model.predict_proba(X)
    risks = model.classify_risk(X)
    df['churn_probability'] = probs
    df['risk_category'] = risks
    df.to_csv('reports/customer_predictions.csv', index=False)
    
    feature_importance = model.get_feature_importance()
    print("\nTop features:")
    print(feature_importance.head(10).to_string(index=False))
    feature_importance.to_csv('reports/feature_importance.csv', index=False)
    
    analyzer = ChurnAnalyzer(df, probs, risks)
    insights = analyzer.generate_business_insights(feature_importance)
    print("\n" + insights)
    with open('reports/business_insights.txt', 'w') as f:
        f.write(insights)
    
    print("\nSaving charts...")
    viz = ChurnVisualizer(df)
    viz.create_full_report(feature_importance, model.get_confusion_matrix(X_test, y_test), metrics)
    
    print("\n" + "=" * 60)
    print("All done! Run: streamlit run dashboard.py")
    print("=" * 60)


if __name__ == '__main__':
    main()