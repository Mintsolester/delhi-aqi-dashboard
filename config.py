"""
Configuration Management
Central configuration for all modules with environment-based settings
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Application Configuration"""
    
    # Base Paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    MODEL_DIR = BASE_DIR / "models"
    LOGS_DIR = BASE_DIR / "logs"
    
    # Create directories if they don't exist
    for directory in [DATA_DIR, MODEL_DIR, LOGS_DIR]:
        directory.mkdir(exist_ok=True)
    
    # API Keys (Load from environment or Streamlit secrets)
    OPENAQ_API_KEY = os.getenv("OPENAQ_API_KEY", "")
    OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY", "")
    SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY", "")
    TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
    TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
    FIREBASE_CREDENTIALS_PATH = os.getenv("FIREBASE_CREDENTIALS_PATH", "")
    
    # Database Configuration
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///delhi_aqi.db")
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # Geographic Configuration
    DELHI_BBOX = {
        'min_lat': 28.4041,
        'max_lat': 28.8833,
        'min_lon': 76.8388,
        'max_lon': 77.3465
    }
    
    GURGAON_BBOX = {
        'min_lat': 28.4089,
        'max_lat': 28.5020,
        'min_lon': 76.9730,
        'max_lon': 77.1133
    }
    
    # Data Collection Parameters
    HISTORICAL_YEARS = 10  # Fetch last 10 years of data
    DATA_UPDATE_INTERVAL_HOURS = 1  # Update every hour
    
    # Model Configuration
    LSTM_SEQUENCE_LENGTH = 168  # 7 days of hourly data
    LSTM_FORECAST_HORIZON = 24  # Predict next 24 hours
    MODEL_RETRAIN_DAYS = 7  # Retrain weekly
    MODEL_VERSION_DIR = MODEL_DIR / "versions"
    
    # Model Hyperparameters
    LSTM_CONFIG = {
        'units': [128, 64, 32],
        'dropout': 0.2,
        'learning_rate': 0.001,
        'batch_size': 16,  # Reduced from 32 for 4GB GPU (GTX 1650)
        'epochs': 50,
        'validation_split': 0.2
    }
    
    # Pollutant Parameters
    POLLUTANTS = ['pm25', 'pm10', 'no2', 'so2', 'o3', 'co']
    PRIMARY_POLLUTANT = 'pm25'  # Focus pollutant for forecasting
    
    # AQI Categories (India CPCB Standards)
    AQI_CATEGORIES = {
        'Good': {'range': (0, 50), 'color': '#00e400', 'icon': '😊'},
        'Satisfactory': {'range': (51, 100), 'color': '#ffff00', 'icon': '🙂'},
        'Moderate': {'range': (101, 200), 'color': '#ff7e00', 'icon': '😐'},
        'Poor': {'range': (201, 300), 'color': '#ff0000', 'icon': '😷'},
        'Very Poor': {'range': (301, 400), 'color': '#8f3f97', 'icon': '🤢'},
        'Severe': {'range': (401, 500), 'color': '#7e0023', 'icon': '☠️'}
    }
    
    # Health Advisory Settings
    VULNERABLE_GROUPS = ['elderly', 'children', 'asthmatic', 'heart_disease', 'pregnant']
    ALERT_THRESHOLD_AQI = 200  # Send alerts when AQI > 200
    
    # Notification Settings
    EMAIL_FROM = os.getenv("EMAIL_FROM", "noreply@delhiaqi.app")
    SMS_FROM = os.getenv("SMS_FROM", "+1234567890")
    
    # Cache Settings
    CACHE_TTL_MINUTES = 30
    
    # MLflow Tracking
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    MLFLOW_EXPERIMENT_NAME = "delhi_aqi_forecasting"
    
    # Feature Engineering
    WEATHER_FEATURES = [
        'temperature_c', 'humidity_percent', 'wind_speed_mps',
        'pressure_hpa', 'cloud_cover', 'precipitation_mm'
    ]
    
    TEMPORAL_FEATURES = [
        'hour', 'day_of_week', 'month', 'is_weekend',
        'is_holiday', 'is_diwali', 'is_crop_burning_season'
    ]
    
    # Event Markers (for historical context)
    SPECIAL_EVENTS = {
        'diwali': 'October-November (varies)',
        'crop_burning': 'October-December',
        'winter_inversion': 'December-February',
        'holi': 'March',
        'construction_ban': 'October-January'
    }
    
    # Data Privacy
    ANONYMIZE_USER_DATA = True
    DATA_RETENTION_DAYS = 365
    
    # UI Configuration
    MAP_DEFAULT_CENTER = [28.6139, 77.2090]  # Delhi
    MAP_DEFAULT_ZOOM = 11
    
    # Social Sharing
    WHATSAPP_SHARE_TEMPLATE = "🌍 Current AQI in {location}: {aqi} ({category}) - Check real-time air quality: {url}"
    
    @classmethod
    def get_aqi_category(cls, aqi_value):
        """Get AQI category for a given value"""
        for category, info in cls.AQI_CATEGORIES.items():
            min_val, max_val = info['range']
            if min_val <= aqi_value <= max_val:
                return {
                    'category': category,
                    'color': info['color'],
                    'icon': info['icon']
                }
        return {
            'category': 'Hazardous',
            'color': '#7e0023',
            'icon': '☠️'
        }
    
    @classmethod
    def is_crop_burning_season(cls, date):
        """Check if date falls in crop burning season"""
        month = date.month
        return month in [10, 11, 12]
    
    @classmethod
    def is_winter_season(cls, date):
        """Check if date falls in winter"""
        month = date.month
        return month in [12, 1, 2]

# Global config instance
config = Config()
