"""
Analytics Module
Implements proof-of-concept machine learning model
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def prepare_features(integrated_df, target_parameter='pm25'):
    """
    Prepare feature matrix and target vector for ML
    """
    df_filtered = integrated_df[integrated_df['parameter'] == target_parameter].copy()
    
    if len(df_filtered) == 0:
        return None, None, None
    
    feature_columns = [
        'temperature_c',
        'humidity_percent',
        'wind_speed_mps',
        'pressure_hpa',
        'hour'
    ]
    
    available_features = [col for col in feature_columns if col in df_filtered.columns]
    
    if len(available_features) == 0:
        return None, None, None
    
    df_clean = df_filtered[available_features + ['value']].dropna()
    
    if len(df_clean) < 5:
        return None, None, None
    
    X = df_clean[available_features].values
    y = df_clean['value'].values
    
    return X, y, available_features

def train_regression_model(X, y, test_size=0.2, random_state=42):
    """
    Train a linear regression model
    """
    if len(X) < 10:
        X_train, X_test = X, X
        y_train, y_test = y, y
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred_test = model.predict(X_test)
    
    metrics = {
        'r2_score': r2_score(y_test, y_pred_test),
        'mae': mean_absolute_error(y_test, y_pred_test),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test))
    }
    
    return model, metrics

def generate_insights(model, X, feature_names):
    """
    Generate interpretable insights from the model
    """
    coefficients = model.coef_
    intercept = model.intercept_
    
    insights = {
        'features': feature_names,
        'coefficients': coefficients.tolist(),
        'intercept': float(intercept),
        'interpretations': []
    }
    
    for feature, coef in zip(feature_names, coefficients):
        direction = "positive" if coef > 0 else "negative"
        
        interpretation = {
            'feature': feature,
            'coefficient': float(coef),
            'relationship': direction
        }
        
        insights['interpretations'].append(interpretation)
    
    insights['interpretations'].sort(key=lambda x: abs(x['coefficient']), reverse=True)
    
    return insights
