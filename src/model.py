# churn prediction model
# supports random forest, gradient boosting, and logistic regression

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
import joblib
from typing import Dict, List, Tuple, Optional, Any


class ChurnModel:
    """wrapper around sklearn classifiers for churn prediction"""
    
    # these thresholds seemed reasonable for our use case
    HIGH_RISK_THRESHOLD = 0.75
    MEDIUM_RISK_THRESHOLD = 0.40
    
    def __init__(self, model_type: str = 'random_forest'):
        self.model_type = model_type
        self.model = None
        self.feature_names = None
        self.is_fitted = False
        
    def _create_model(self) -> Any:
        # factory method to create the right sklearn model
        if self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100, max_depth=10, min_samples_split=5,
                class_weight='balanced', random_state=42, n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
            )
        elif self.model_type == 'logistic_regression':
            return LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
        else:
            raise ValueError(f"unknown model type: {self.model_type}")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              feature_names: Optional[List[str]] = None) -> 'ChurnModel':
        self.model = self._create_model()
        self.model.fit(X_train, y_train)
        self.feature_names = feature_names
        self.is_fitted = True
        return self
    
    def train_with_tuning(self, X_train: np.ndarray, y_train: np.ndarray,
                          feature_names: Optional[List[str]] = None, cv: int = 5) -> 'ChurnModel':
        if self.model_type == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5, 10]
            }
            base_model = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)
        elif self.model_type == 'gradient_boosting':
            param_grid = {
                'n_estimators': [50, 100, 150],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.05, 0.1, 0.2]
            }
            base_model = GradientBoostingClassifier(random_state=42)
        else:
            return self.train(X_train, y_train, feature_names)
        
        grid_search = GridSearchCV(base_model, param_grid, cv=cv, scoring='f1', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        self.feature_names = feature_names
        self.is_fitted = True
        self.best_params = grid_search.best_params_
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model is not fitted.")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # returns probability of churn (class 1)
        if not self.is_fitted:
            raise ValueError("Model is not fitted.")
        return self.model.predict_proba(X)[:, 1]
    
    def classify_risk(self, X: np.ndarray) -> List[str]:
        # bucket customers into risk categories based on churn probability
        probabilities = self.predict_proba(X)
        risk_categories = []
        for prob in probabilities:
            if prob > self.HIGH_RISK_THRESHOLD:
                risk_categories.append('High Risk')
            elif prob >= self.MEDIUM_RISK_THRESHOLD:
                risk_categories.append('Medium Risk')
            else:
                risk_categories.append('Low Risk')
        return risk_categories
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba)
        }
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict[str, Tuple[float, float]]:
        model = self._create_model() if not self.is_fitted else self.model
        results = {}
        for scoring in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
            results[scoring] = (scores.mean(), scores.std())
        return results
    
    def get_feature_importance(self) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Model is not fitted.")
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_[0])
        else:
            raise ValueError("Model does not support feature importance")
        
        feature_names = self.feature_names or [f'feature_{i}' for i in range(len(importances))]
        return pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
    
    def get_classification_report(self, X_test: np.ndarray, y_test: np.ndarray) -> str:
        y_pred = self.predict(X_test)
        return classification_report(y_test, y_pred, target_names=['No Churn', 'Churn'])
    
    def get_confusion_matrix(self, X_test: np.ndarray, y_test: np.ndarray) -> np.ndarray:
        y_pred = self.predict(X_test)
        return confusion_matrix(y_test, y_pred)
    
    def save(self, filepath: str = 'models/churn_model.pkl'):
        if not self.is_fitted:
            raise ValueError("Model is not fitted.")
        joblib.dump({
            'model': self.model,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }, filepath)
    
    @classmethod
    def load(cls, filepath: str = 'models/churn_model.pkl') -> 'ChurnModel':
        model_data = joblib.load(filepath)
        instance = cls(model_type=model_data['model_type'])
        instance.model = model_data['model']
        instance.feature_names = model_data['feature_names']
        instance.is_fitted = model_data['is_fitted']
        return instance