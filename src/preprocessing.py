# data preprocessing utils

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from typing import Tuple, List, Optional


def load_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """basic data cleaning - handle missing values, remove dupes"""
    df = df.copy()
    
    if 'InternetService' in df.columns:
        df['InternetService'] = df['InternetService'].fillna('None')
    
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna('Unknown')
    
    for col in df.select_dtypes(include=['number']).columns:
        df[col] = df[col].fillna(df[col].median())
    
    df = df.drop_duplicates()
    return df


def prepare_features(
    df: pd.DataFrame,
    target_col: str = 'Churn',
    exclude_cols: Optional[List[str]] = None,
    scaler: Optional[StandardScaler] = None,
    fit_scaler: bool = True
) -> Tuple[np.ndarray, np.ndarray, StandardScaler, List[str]]:
    """one-hot encode categoricals, scale numerics, return X and y"""
    df = df.copy()
    
    if exclude_cols is None:
        exclude_cols = ['CustomerID']
    
    if target_col in df.columns:
        y = df[target_col].map({'Yes': 1, 'No': 0}).values
        df = df.drop(columns=[target_col])
    else:
        y = None
    
    for col in exclude_cols:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    feature_names = df.columns.tolist()
    X = df.values
    
    if scaler is None:
        scaler = StandardScaler()
    
    if fit_scaler:
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)
    
    return X, y, scaler, feature_names


def prepare_single_customer(
    customer_data: dict,
    scaler: StandardScaler,
    feature_names: List[str]
) -> np.ndarray:
    """transform a single customer dict into model-ready features"""
    df = pd.DataFrame([customer_data])
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    df = pd.get_dummies(df, columns=categorical_cols)
    
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    
    df = df[feature_names]
    X = scaler.transform(df.values)
    return X


def save_preprocessing_artifacts(scaler, feature_names, filepath_prefix='models/'):
    # save scaler and feature names so we can use them later
    joblib.dump(scaler, f'{filepath_prefix}scaler.pkl')
    joblib.dump(feature_names, f'{filepath_prefix}feature_names.pkl')


def load_preprocessing_artifacts(filepath_prefix='models/'):
    scaler = joblib.load(f'{filepath_prefix}scaler.pkl')
    feature_names = joblib.load(f'{filepath_prefix}feature_names.pkl')
    return scaler, feature_names