"""
Model Monitoring and Tracking
Tracks predictions vs actuals, computes accuracy, shows model progress
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
from pathlib import Path
import logging

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTracker:
    """
    Tracks model predictions against actual values
    Computes accuracy metrics and maintains model performance history
    """
    
    def __init__(self, tracking_file: str = "model_tracking.json"):
        """
        Initialize model tracker
        
        Args:
            tracking_file: JSON file to store tracking data
        """
        self.tracking_file = config.DATA_DIR / tracking_file
        self.predictions_log = []
        self.actuals_log = []
        self._load_tracking_data()
    
    def _load_tracking_data(self):
        """Load existing tracking data"""
        if self.tracking_file.exists():
            try:
                with open(self.tracking_file, 'r') as f:
                    data = json.load(f)
                    self.predictions_log = data.get('predictions', [])
                    self.actuals_log = data.get('actuals', [])
                logger.info(f"Loaded {len(self.predictions_log)} tracking records")
            except Exception as e:
                logger.error(f"Error loading tracking data: {str(e)}")
        else:
            logger.info("No existing tracking data found")
    
    def _save_tracking_data(self):
        """Save tracking data to file"""
        try:
            data = {
                'predictions': self.predictions_log,
                'actuals': self.actuals_log,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.tracking_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info("Tracking data saved")
        except Exception as e:
            logger.error(f"Error saving tracking data: {str(e)}")
    
    def log_prediction(self,
                      location: str,
                      timestamp: datetime,
                      predicted_values: List[float],
                      model_version: str,
                      forecast_horizon: int = 24):
        """
        Log model prediction
        
        Args:
            location: Location name
            timestamp: Prediction timestamp
            predicted_values: List of predicted AQI values
            model_version: Model version identifier
            forecast_horizon: Number of hours predicted
        """
        prediction_record = {
            'location': location,
            'prediction_timestamp': timestamp.isoformat(),
            'forecast_timestamps': [
                (timestamp + timedelta(hours=i)).isoformat()
                for i in range(1, forecast_horizon + 1)
            ],
            'predicted_values': predicted_values,
            'model_version': model_version,
            'logged_at': datetime.now().isoformat()
        }
        
        self.predictions_log.append(prediction_record)
        self._save_tracking_data()
        
        logger.info(f"Logged prediction for {location} at {timestamp}")
    
    def log_actual(self,
                  location: str,
                  timestamp: datetime,
                  actual_value: float):
        """
        Log actual observed value
        
        Args:
            location: Location name
            timestamp: Observation timestamp
            actual_value: Actual AQI value
        """
        actual_record = {
            'location': location,
            'timestamp': timestamp.isoformat(),
            'actual_value': actual_value,
            'logged_at': datetime.now().isoformat()
        }
        
        self.actuals_log.append(actual_record)
        self._save_tracking_data()
    
    def get_prediction_accuracy(self,
                               location: Optional[str] = None,
                               days_back: int = 7) -> Dict:
        """
        Calculate prediction accuracy for recent period
        
        Args:
            location: Optional location filter
            days_back: Number of days to analyze
            
        Returns:
            Dict with accuracy metrics
        """
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        # Filter predictions
        predictions_df = pd.DataFrame(self.predictions_log)
        actuals_df = pd.DataFrame(self.actuals_log)
        
        if len(predictions_df) == 0 or len(actuals_df) == 0:
            return {
                'status': 'insufficient_data',
                'message': 'Not enough data to compute accuracy'
            }
        
        # Convert timestamps
        predictions_df['prediction_timestamp'] = pd.to_datetime(
            predictions_df['prediction_timestamp']
        )
        actuals_df['timestamp'] = pd.to_datetime(actuals_df['timestamp'])
        
        # Filter by date
        predictions_df = predictions_df[
            predictions_df['prediction_timestamp'] >= cutoff_date
        ]
        actuals_df = actuals_df[actuals_df['timestamp'] >= cutoff_date]
        
        # Filter by location if specified
        if location:
            predictions_df = predictions_df[predictions_df['location'] == location]
            actuals_df = actuals_df[actuals_df['location'] == location]
        
        # Match predictions with actuals
        matched_pairs = []
        
        for _, pred_row in predictions_df.iterrows():
            forecast_timestamps = pred_row['forecast_timestamps']
            predicted_values = pred_row['predicted_values']
            
            for i, forecast_ts in enumerate(forecast_timestamps):
                forecast_dt = pd.to_datetime(forecast_ts)
                
                # Find matching actual
                actual_match = actuals_df[
                    (actuals_df['location'] == pred_row['location']) &
                    (abs((actuals_df['timestamp'] - forecast_dt).dt.total_seconds()) < 3600)  # Within 1 hour
                ]
                
                if len(actual_match) > 0:
                    matched_pairs.append({
                        'predicted': predicted_values[i],
                        'actual': actual_match.iloc[0]['actual_value'],
                        'timestamp': forecast_ts,
                        'location': pred_row['location']
                    })
        
        if len(matched_pairs) == 0:
            return {
                'status': 'no_matches',
                'message': 'No matching prediction-actual pairs found'
            }
        
        # Calculate metrics
        matched_df = pd.DataFrame(matched_pairs)
        
        y_true = matched_df['actual'].values
        y_pred = matched_df['predicted'].values
        
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # Calculate percentage error
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # Accuracy percentage (inverse of MAPE, capped at 100%)
        accuracy_pct = max(0, min(100, 100 - mape))
        
        metrics = {
            'status': 'success',
            'n_comparisons': len(matched_pairs),
            'mae': round(mae, 2),
            'rmse': round(rmse, 2),
            'r2': round(r2, 3),
            'mape': round(mape, 2),
            'accuracy_percentage': round(accuracy_pct, 1),
            'period_days': days_back,
            'location': location or 'All locations',
            'computed_at': datetime.now().isoformat()
        }
        
        return metrics
    
    def get_model_progress(self) -> Dict:
        """
        Get model improvement progress over time
        
        Returns:
            Dict with progress metrics
        """
        if len(self.predictions_log) == 0:
            return {
                'status': 'no_data',
                'message': 'Model has not made predictions yet'
            }
        
        # Calculate accuracy for different time periods
        accuracy_7d = self.get_prediction_accuracy(days_back=7)
        accuracy_30d = self.get_prediction_accuracy(days_back=30)
        
        # Get first and last prediction dates
        predictions_df = pd.DataFrame(self.predictions_log)
        predictions_df['logged_at'] = pd.to_datetime(predictions_df['logged_at'])
        
        first_prediction = predictions_df['logged_at'].min()
        last_prediction = predictions_df['logged_at'].max()
        
        # Model versions used
        model_versions = predictions_df['model_version'].unique().tolist()
        
        progress = {
            'first_prediction_date': first_prediction.strftime('%Y-%m-%d'),
            'last_prediction_date': last_prediction.strftime('%Y-%m-%d'),
            'total_predictions': len(predictions_df),
            'model_versions': model_versions,
            'latest_version': model_versions[-1] if model_versions else 'Unknown',
            'accuracy_7_days': accuracy_7d,
            'accuracy_30_days': accuracy_30d,
            'status': 'improving' if accuracy_7d.get('accuracy_percentage', 0) > 
                                    accuracy_30d.get('accuracy_percentage', 0) else 'stable',
            'computed_at': datetime.now().isoformat()
        }
        
        return progress
    
    def get_prediction_vs_actual_data(self,
                                     location: Optional[str] = None,
                                     days_back: int = 7) -> pd.DataFrame:
        """
        Get matched prediction vs actual data for visualization
        
        Args:
            location: Optional location filter
            days_back: Number of days to retrieve
            
        Returns:
            DataFrame with predictions and actuals
        """
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        predictions_df = pd.DataFrame(self.predictions_log)
        actuals_df = pd.DataFrame(self.actuals_log)
        
        if len(predictions_df) == 0 or len(actuals_df) == 0:
            return pd.DataFrame()
        
        # Convert timestamps
        predictions_df['prediction_timestamp'] = pd.to_datetime(
            predictions_df['prediction_timestamp']
        )
        actuals_df['timestamp'] = pd.to_datetime(actuals_df['timestamp'])
        
        # Filter
        predictions_df = predictions_df[
            predictions_df['prediction_timestamp'] >= cutoff_date
        ]
        actuals_df = actuals_df[actuals_df['timestamp'] >= cutoff_date]
        
        if location:
            predictions_df = predictions_df[predictions_df['location'] == location]
            actuals_df = actuals_df[actuals_df['location'] == location]
        
        # Expand predictions into individual forecasts
        forecast_rows = []
        
        for _, pred_row in predictions_df.iterrows():
            for i, forecast_ts in enumerate(pred_row['forecast_timestamps']):
                forecast_rows.append({
                    'timestamp': pd.to_datetime(forecast_ts),
                    'predicted': pred_row['predicted_values'][i],
                    'location': pred_row['location'],
                    'model_version': pred_row['model_version']
                })
        
        forecast_df = pd.DataFrame(forecast_rows)
        
        # Merge with actuals
        if len(forecast_df) > 0:
            merged = pd.merge_asof(
                forecast_df.sort_values('timestamp'),
                actuals_df[['timestamp', 'actual_value', 'location']].sort_values('timestamp'),
                on='timestamp',
                by='location',
                tolerance=pd.Timedelta('1hour'),
                direction='nearest'
            )
            
            merged = merged.rename(columns={'actual_value': 'actual'})
            return merged
        
        return pd.DataFrame()
    
    def generate_accuracy_badge(self) -> Dict:
        """
        Generate badge data for displaying model accuracy
        
        Returns:
            Dict with badge information
        """
        accuracy = self.get_prediction_accuracy(days_back=7)
        
        if accuracy.get('status') != 'success':
            return {
                'accuracy': 'N/A',
                'color': 'gray',
                'label': 'Model Accuracy',
                'status': 'training'
            }
        
        acc_pct = accuracy['accuracy_percentage']
        
        # Determine badge color
        if acc_pct >= 85:
            color = 'green'
            status = 'excellent'
        elif acc_pct >= 70:
            color = 'yellow'
            status = 'good'
        elif acc_pct >= 50:
            color = 'orange'
            status = 'fair'
        else:
            color = 'red'
            status = 'improving'
        
        return {
            'accuracy': f"{acc_pct:.1f}%",
            'color': color,
            'label': 'Model Accuracy (7 days)',
            'status': status,
            'n_comparisons': accuracy['n_comparisons'],
            'last_updated': accuracy['computed_at']
        }


if __name__ == "__main__":
    # Example usage
    tracker = ModelTracker()
    
    # Simulate logging predictions
    location = "Anand Vihar"
    now = datetime.now()
    
    # Log a prediction
    predicted_values = [150 + i * 2 for i in range(24)]
    tracker.log_prediction(
        location=location,
        timestamp=now,
        predicted_values=predicted_values,
        model_version="20250102_120000",
        forecast_horizon=24
    )
    
    # Log some actuals
    for i in range(24):
        tracker.log_actual(
            location=location,
            timestamp=now + timedelta(hours=i+1),
            actual_value=148 + i * 2 + np.random.normal(0, 5)
        )
    
    # Get accuracy
    accuracy = tracker.get_prediction_accuracy(location=location, days_back=1)
    print(f"\nModel Accuracy:")
    print(json.dumps(accuracy, indent=2))
    
    # Get progress
    progress = tracker.get_model_progress()
    print(f"\nModel Progress:")
    print(json.dumps(progress, indent=2))
    
    # Get badge
    badge = tracker.generate_accuracy_badge()
    print(f"\nAccuracy Badge:")
    print(json.dumps(badge, indent=2))
