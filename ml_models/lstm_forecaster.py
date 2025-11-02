"""
LSTM-based AQI Forecasting Model - FIXED SCALER SAVING
Next-day prediction with ward/location-level granularity
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json
import joblib

from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import mlflow
import mlflow.keras

import sys
from pathlib import Path
# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_gpu_availability():
    """Check if GPU is available and print GPU information"""
    import tensorflow as tf
    
    print("\n" + "="*70)
    print("GPU AVAILABILITY CHECK")
    print("="*70)
    
    # List physical devices
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        print(f"✅ GPU DETECTED: {len(gpus)} GPU(s) available")
        for i, gpu in enumerate(gpus):
            print(f"   GPU {i}: {gpu.name}")
            # Get GPU details
            try:
                gpu_details = tf.config.experimental.get_device_details(gpu)
                print(f"   Compute Capability: {gpu_details.get('compute_capability', 'N/A')}")
            except:
                pass
        
        # Check if TensorFlow is built with CUDA
        print(f"\n✅ CUDA Support: {tf.test.is_built_with_cuda()}")
        
        # Memory growth to avoid allocating all GPU memory at once
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✅ Memory growth enabled for {gpu.name}")
            except RuntimeError as e:
                print(f"⚠️  Memory growth setting: {e}")
    else:
        print("❌ NO GPU DETECTED - Training will use CPU only")
        print("   This will be significantly slower for large models.")
        print("\n   Troubleshooting:")
        print("   1. Ensure you installed: pip install tensorflow[and-cuda]")
        print("   2. Check NVIDIA drivers: nvidia-smi")
        print("   3. Verify CUDA compatibility with your TensorFlow version")
    
    print(f"\nTensorFlow version: {tf.__version__}")
    print("="*70 + "\n")
    
    return len(gpus) > 0


class AQIForecaster:
    """
    LSTM-based forecasting model for next-day AQI prediction
    Supports multi-location, multi-pollutant forecasting
    """
    
    def __init__(self, 
                 sequence_length: int = 168,  # 7 days of hourly data
                 forecast_horizon: int = 24):  # Predict next 24 hours
        """
        Initialize forecaster
        
        Args:
            sequence_length: Number of past timesteps to use
            forecast_horizon: Number of future timesteps to predict
        """
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.model = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.feature_columns = None
        self.model_version = None
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features for LSTM model
        
        Args:
            df: Raw dataset with AQI, weather, and events
            
        Returns:
            Feature-engineered DataFrame
        """
        logger.info("Preparing features for LSTM")
        
        df = df.copy()
        
        # Ensure datetime index
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
        
        # Sort by index
        df = df.sort_index()
        
        # Select feature columns
        feature_cols = []
        
        # Weather features
        for col in config.WEATHER_FEATURES:
            if col in df.columns:
                feature_cols.append(col)
        
        # Temporal features
        for col in config.TEMPORAL_FEATURES:
            if col in df.columns:
                feature_cols.append(col)
        
        # Lagged AQI features (past pollutant values)
        if 'value' in df.columns:
            for lag in [1, 2, 3, 6, 12, 24]:
                df[f'aqi_lag_{lag}h'] = df['value'].shift(lag)
                feature_cols.append(f'aqi_lag_{lag}h')
        
        # Rolling statistics
        if 'value' in df.columns:
            df['aqi_rolling_mean_24h'] = df['value'].rolling(window=24).mean()
            df['aqi_rolling_std_24h'] = df['value'].rolling(window=24).std()
            df['aqi_rolling_max_24h'] = df['value'].rolling(window=24).max()
            feature_cols.extend(['aqi_rolling_mean_24h', 'aqi_rolling_std_24h', 'aqi_rolling_max_24h'])
        
        # Weather interaction features
        if 'temperature_c' in df.columns and 'humidity_percent' in df.columns:
            df['temp_humidity_interaction'] = df['temperature_c'] * df['humidity_percent']
            feature_cols.append('temp_humidity_interaction')
        
        # Remove rows with NaN (created by lagging/rolling)
        df = df.dropna()
        
        self.feature_columns = feature_cols
        
        logger.info(f"Feature engineering complete. Total features: {len(feature_cols)}")
        
        return df
    
    def create_sequences(self, data: np.ndarray, target: np.ndarray):
        """
        Create sequences for LSTM training
        
        Args:
            data: Feature array
            target: Target array
            
        Returns:
            X, y sequences
        """
        X, y = [], []
        
        for i in range(self.sequence_length, len(data) - self.forecast_horizon):
            X.append(data[i - self.sequence_length:i])
            y.append(target[i:i + self.forecast_horizon])
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape: tuple) -> keras.Model:
        """
        Build LSTM architecture
        
        Args:
            input_shape: Shape of input sequences (sequence_length, n_features)
            
        Returns:
            Compiled Keras model
        """
        logger.info(f"Building LSTM model with input shape: {input_shape}")
        
        model = Sequential([
            # First LSTM layer
            LSTM(config.LSTM_CONFIG['units'][0], 
                 return_sequences=True,
                 input_shape=input_shape),
            Dropout(config.LSTM_CONFIG['dropout']),
            BatchNormalization(),
            
            # Second LSTM layer
            LSTM(config.LSTM_CONFIG['units'][1], return_sequences=True),
            Dropout(config.LSTM_CONFIG['dropout']),
            BatchNormalization(),
            
            # Third LSTM layer
            LSTM(config.LSTM_CONFIG['units'][2], return_sequences=False),
            Dropout(config.LSTM_CONFIG['dropout']),
            
            # Dense layers for output
            Dense(64, activation='relu'),
            Dropout(0.1),
            Dense(self.forecast_horizon)  # Predict next 24 hours
        ])
        
        optimizer = Adam(learning_rate=config.LSTM_CONFIG['learning_rate'])
        
        model.compile(
            optimizer=optimizer,
            loss='huber',  # Robust to outliers
            metrics=['mae', 'mse']
        )
        
        logger.info(f"Model architecture:\n{model.summary()}")
        
        return model
    
    def train(self, 
              df: pd.DataFrame,
              target_column: str = 'value',
              location_filter: str = None,
              validation_split: float = 0.2):
        """
        Train LSTM model
        
        Args:
            df: Training dataset
            target_column: Column to predict
            location_filter: Optional location filter
            validation_split: Validation data proportion
        """
        logger.info("Starting model training")
        
        # Start MLflow run
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
        
        with mlflow.start_run():
            # Filter by location if specified
            if location_filter:
                df = df[df['location_name'] == location_filter]
            
            # Prepare features
            df = self.prepare_features(df)
            
            # Extract features and target
            X_data = df[self.feature_columns].values
            y_data = df[target_column].values
            
            # Scale data
            X_scaled = self.scaler_X.fit_transform(X_data)
            y_scaled = self.scaler_y.fit_transform(y_data.reshape(-1, 1)).flatten()
            
            # Create sequences
            X_seq, y_seq = self.create_sequences(X_scaled, y_scaled)
            
            logger.info(f"Sequence shapes - X: {X_seq.shape}, y: {y_seq.shape}")
            
            # Train-validation split
            split_idx = int(len(X_seq) * (1 - validation_split))
            X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
            y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]
            
            # Build model
            self.model = self.build_model(input_shape=(self.sequence_length, X_data.shape[1]))
            
            # Callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True,
                    verbose=1
                ),
                ModelCheckpoint(
                    str(config.MODEL_DIR / 'best_model.keras'),  # Changed to .keras
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    verbose=1
                )
            ]
            
            # Log parameters
            mlflow.log_params(config.LSTM_CONFIG)
            mlflow.log_param("sequence_length", self.sequence_length)
            mlflow.log_param("forecast_horizon", self.forecast_horizon)
            mlflow.log_param("n_features", len(self.feature_columns))
            
            # Train
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=config.LSTM_CONFIG['epochs'],
                batch_size=config.LSTM_CONFIG['batch_size'],
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluate
            y_pred = self.model.predict(X_val)
            
            # Inverse transform for metrics
            y_val_original = self.scaler_y.inverse_transform(y_val)
            y_pred_original = self.scaler_y.inverse_transform(y_pred)
            
            # Calculate metrics
            mae = mean_absolute_error(y_val_original, y_pred_original)
            mse = mean_squared_error(y_val_original, y_pred_original)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_val_original, y_pred_original)
            
            metrics = {
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'r2': r2
            }
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Save model with MLflow
            mlflow.keras.log_model(self.model, "model")
            
            # Save model version info
            self.model_version = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            logger.info(f"Training complete. Metrics: {metrics}")
            
            # CRITICAL FIX: Save scalers alongside best_model.h5
            self.save_model_and_scalers()
            
            # Also save versioned copy
            self.save_model()
            
            return history, metrics
    
    def save_model_and_scalers(self):
        """Save scalers in the same directory as best_model.h5 for easy loading"""
        model_dir = config.MODEL_DIR
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save scalers alongside best_model.h5
        joblib.dump(self.scaler_X, model_dir / 'scaler_X.pkl')
        joblib.dump(self.scaler_y, model_dir / 'scaler_y.pkl')
        
        # Save feature columns
        with open(model_dir / 'feature_columns.json', 'w') as f:
            json.dump({
                'feature_columns': self.feature_columns,
                'sequence_length': self.sequence_length,
                'forecast_horizon': self.forecast_horizon
            }, f, indent=2)
        
        logger.info(f"✅ Scalers saved to {model_dir}")
    
    def predict(self, 
                recent_data: pd.DataFrame,
                steps_ahead: int = 24) -> np.ndarray:
        """
        Make predictions for next N hours
        
        Args:
            recent_data: Recent data (at least sequence_length hours)
            steps_ahead: Number of hours to forecast
            
        Returns:
            Predicted AQI values
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        if self.scaler_X is None or self.scaler_y is None:
            raise ValueError("Scalers not fitted. Train the model or load scalers.")
        
        # Prepare features
        recent_data = self.prepare_features(recent_data)
        
        # Get last sequence_length rows
        X_recent = recent_data[self.feature_columns].values[-self.sequence_length:]
        
        # Scale
        X_scaled = self.scaler_X.transform(X_recent)
        
        # Reshape for LSTM
        X_input = X_scaled.reshape(1, self.sequence_length, -1)
        
        # Predict
        y_pred_scaled = self.model.predict(X_input, verbose=0)
        
        # Inverse transform
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        
        return y_pred.flatten()[:steps_ahead]
    
    def save_model(self, version: str = None):
        """Save model, scalers, and metadata"""
        
        if version is None:
            version = self.model_version or datetime.now().strftime('%Y%m%d_%H%M%S')
        
        version_dir = config.MODEL_VERSION_DIR / version
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Save Keras model
        model_path = version_dir / 'lstm_model.h5'
        self.model.save(str(model_path))
        
        # Save scalers
        joblib.dump(self.scaler_X, version_dir / 'scaler_X.pkl')
        joblib.dump(self.scaler_y, version_dir / 'scaler_y.pkl')
        
        # Save metadata
        metadata = {
            'version': version,
            'created_at': datetime.now().isoformat(),
            'sequence_length': self.sequence_length,
            'forecast_horizon': self.forecast_horizon,
            'feature_columns': self.feature_columns,
            'model_config': config.LSTM_CONFIG
        }
        
        with open(version_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {version_dir}")
    
    def load_model(self, version: str = 'latest'):
        """Load model, scalers, and metadata"""
        
        if version == 'latest':
            # Find latest version
            versions = sorted(config.MODEL_VERSION_DIR.glob('*'))
            if not versions:
                raise FileNotFoundError("No saved models found")
            version_dir = versions[-1]
        else:
            version_dir = config.MODEL_VERSION_DIR / version
        
        if not version_dir.exists():
            raise FileNotFoundError(f"Model version {version} not found")
        
        # Load model
        model_path = version_dir / 'lstm_model.h5'
        self.model = load_model(str(model_path))
        
        # Load scalers
        self.scaler_X = joblib.load(version_dir / 'scaler_X.pkl')
        self.scaler_y = joblib.load(version_dir / 'scaler_y.pkl')
        
        # Load metadata
        with open(version_dir / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        
        self.sequence_length = metadata['sequence_length']
        self.forecast_horizon = metadata['forecast_horizon']
        self.feature_columns = metadata['feature_columns']
        self.model_version = metadata['version']
        
        logger.info(f"Model loaded from {version_dir}")
        
        return metadata


if __name__ == "__main__":
    # Check GPU availability first
    has_gpu = check_gpu_availability()
    
    # Example usage
    from data_pipeline import HistoricalDataPipeline
    
    # Load data
    data_path = config.DATA_DIR / "historical_aqi_dataset.csv"
    if data_path.exists():
        print(f"Loading dataset from {data_path}")
        df = pd.read_csv(data_path)
    else:
        print("Dataset not found. Running data pipeline...")
        pipeline = HistoricalDataPipeline(years_back=2)
        df = pipeline.run_full_pipeline()
    
    print(f"Dataset loaded: {len(df)} records")
    
    # Filter for PM2.5
    df_pm25 = df[df['parameter'] == 'pm25'].copy()
    print(f"PM2.5 data: {len(df_pm25)} records")
    
    if len(df_pm25) == 0:
        print("ERROR: No PM2.5 data found in dataset!")
        sys.exit(1)
    
    # MEMORY OPTIMIZATION: Use subset for initial training
    print("\n⚠️  MEMORY OPTIMIZATION:")
    print("   Using last 2 years of data to prevent OOM errors")
    print("   (Full dataset training requires 8GB+ GPU memory)")
    
    # Sample recent data (more relevant for forecasting anyway)
    df_pm25_sorted = df_pm25.sort_values('timestamp')
    rows_to_use = min(len(df_pm25), 17520)  # 2 years * 365 days * 24 hours
    df_pm25_subset = df_pm25_sorted.tail(rows_to_use).copy()
    
    print(f"   Training with {len(df_pm25_subset)} records (last 2 years)")
    
    # Reduce sequence length for memory efficiency
    sequence_length = 72  # 3 days instead of 7 days
    print(f"   Using sequence_length={sequence_length}h (3 days) for memory efficiency\n")
    
    # Train model
    print("="*70)
    print("STARTING MODEL TRAINING")
    print("="*70)
    forecaster = AQIForecaster(sequence_length=sequence_length, forecast_horizon=24)
    history, metrics = forecaster.train(df_pm25_subset, target_column='value')
    
    print(f"\n{'='*70}")
    print("MODEL TRAINING COMPLETE")
    print("="*70)
    print(f"\nModel Performance:")
    for metric, value in metrics.items():
        print(f"  {metric.upper()}: {value:.2f}")
    
    if has_gpu:
        print(f"\n✅ Training completed using GPU acceleration")
    else:
        print(f"\n⚠️  Training completed using CPU (consider enabling GPU for faster training)")
    
    print(f"\n💡 Model and scalers saved to: {config.MODEL_DIR}")
    print(f"   - best_model.h5")
    print(f"   - scaler_X.pkl")
    print(f"   - scaler_y.pkl")
    print(f"   - feature_columns.json")