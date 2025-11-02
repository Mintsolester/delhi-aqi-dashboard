# 🚀 Quick Reference Guide

## Common Commands & Code Snippets

---

## 🎯 **Setup & Installation**

```bash
# Create environment
conda create -n delhi_aqi python=3.11
conda activate delhi_aqi

# Install all dependencies
pip install -r requirements.txt

# Set API keys (Linux/Mac)
export OPENAQ_API_KEY="your_key_here"
export OPENWEATHERMAP_API_KEY="your_key_here"

# Set API keys (Windows)
set OPENAQ_API_KEY=your_key_here
set OPENWEATHERMAP_API_KEY=your_key_here
```

---

## 📊 **Data Collection**

```bash
# Fetch 10 years of historical data
python data_pipeline.py

# Fetch only 2 years (faster for testing)
python -c "
from data_pipeline import HistoricalDataPipeline
pipeline = HistoricalDataPipeline(years_back=2)
dataset = pipeline.run_full_pipeline()
"

# Check data
python -c "
import pandas as pd
df = pd.read_csv('data/historical_aqi_dataset.csv')
print(f'Rows: {len(df)}')
print(f'Locations: {df.location_name.nunique()}')
print(f'Date range: {df.timestamp.min()} to {df.timestamp.max()}')
"
```

---

## 🤖 **Model Training**

```bash
# Train LSTM model
python ml_models/lstm_forecaster.py

# Train with custom parameters
python -c "
from ml_models.lstm_forecaster import AQIForecaster
import pandas as pd

df = pd.read_csv('data/historical_aqi_dataset.csv')
df_pm25 = df[df['parameter'] == 'pm25']

forecaster = AQIForecaster(sequence_length=72, forecast_horizon=24)
history, metrics = forecaster.train(df_pm25)
print(f'Model accuracy: {metrics}')
"

# Load and test model
python -c "
from ml_models.lstm_forecaster import AQIForecaster
import pandas as pd

forecaster = AQIForecaster()
forecaster.load_model('latest')

# Get recent data
df = pd.read_csv('data/historical_aqi_dataset.csv')
recent = df[df['parameter'] == 'pm25'].tail(200)

# Predict
forecast = forecaster.predict(recent, steps_ahead=24)
print(f'Next 24h forecast: {forecast}')
"
```

---

## 🗺️ **Geocoding**

```python
from geocoding import GeocodingService

geocoder = GeocodingService()

# Search by address
location = geocoder.search_location("Connaught Place, Delhi")
print(f"Lat: {location['latitude']}, Lon: {location['longitude']}")
print(f"Ward: {location.get('ward_name')}")

# Search by pincode
location = geocoder.search_location("110001")

# Get location details
details = geocoder.get_location_details(28.6139, 77.2090)
print(details)

# Find nearest station
stations_df = pd.read_csv('data/historical_aqi_dataset.csv')
nearest = geocoder.find_nearest_monitoring_station(
    28.6139, 77.2090, stations_df
)
print(f"Nearest station: {nearest}")
```

---

## 💊 **Health Advisories**

```python
from health_advisor import HealthAdvisor

advisor = HealthAdvisor()

# General advisory
advisory = advisor.get_advisory(aqi_value=180)
print(f"Category: {advisory['category']}")
print(f"Summary: {advisory['summary']}")
for rec in advisory['recommendations']:
    print(f"  - {rec}")

# Vulnerable group advisory
advisory = advisor.get_advisory(
    aqi_value=180,
    vulnerable_groups=['asthmatic', 'elderly']
)

# Print asthma-specific advice
if 'asthmatic' in advisory['vulnerable_groups']:
    advice = advisory['vulnerable_groups']['asthmatic']
    print(f"\nFor Asthmatics:")
    print(f"Risk Level: {advice['risk_level']}")
    for item in advice['specific_advice']:
        print(f"  - {item}")

# Forecast alert
alert = advisor.get_forecast_alert(
    current_aqi=120,
    forecast_aqi=[150, 180, 220, 250, 200, 180] + [150]*18,
    location="Anand Vihar",
    vulnerable_groups=['asthmatic']
)
if alert:
    print(f"\nALERT: {alert['message']}")

# WhatsApp message
msg = advisor.get_whatsapp_message(
    aqi_value=180,
    location="ITO",
    url="https://delhiaqi.app"
)
print(f"\nWhatsApp message:\n{msg}")
```

---

## 🔔 **Notifications**

```python
from notifications import NotificationService

service = NotificationService()

# Subscribe to email alerts
service.subscribe_email(
    email="user@example.com",
    locations=["Anand Vihar", "ITO"],
    threshold_aqi=200,
    vulnerable_groups=['asthmatic']
)

# Subscribe to SMS alerts
service.subscribe_sms(
    phone="+919876543210",
    locations=["Connaught Place"],
    threshold_aqi=150
)

# Send test email
service.send_email_alert(
    to_email="user@example.com",
    subject="Test Alert",
    content="This is a test alert from Delhi AQI Dashboard",
    html_content="<h2>Test Alert</h2><p>This is a test.</p>"
)

# Send forecast alert to all subscribers
service.notify_forecast_alert(
    location="Anand Vihar",
    current_aqi=120,
    forecast_aqi=250,
    peak_time="6:00 PM",
    category="Very Poor"
)

# Get statistics
stats = service.get_subscriber_count()
print(f"Active subscribers: {stats['total_active']}")

# Unsubscribe
service.unsubscribe_email("user@example.com")
```

---

## 📈 **Model Tracking**

```python
from model_tracker import ModelTracker
from datetime import datetime, timedelta
import numpy as np

tracker = ModelTracker()

# Log a prediction
tracker.log_prediction(
    location="Anand Vihar",
    timestamp=datetime.now(),
    predicted_values=[150, 155, 160, 165] + [150]*20,
    model_version="20251102_120000",
    forecast_horizon=24
)

# Log actual values (as they come in)
for i in range(24):
    tracker.log_actual(
        location="Anand Vihar",
        timestamp=datetime.now() + timedelta(hours=i+1),
        actual_value=148 + i*0.5 + np.random.normal(0, 5)
    )

# Get accuracy metrics
accuracy = tracker.get_prediction_accuracy(
    location="Anand Vihar",
    days_back=7
)
print(f"7-day accuracy: {accuracy['accuracy_percentage']:.1f}%")
print(f"MAE: {accuracy['mae']:.2f}")
print(f"RMSE: {accuracy['rmse']:.2f}")

# Get model progress
progress = tracker.get_model_progress()
print(f"Model status: {progress['status']}")
print(f"Latest version: {progress['latest_version']}")

# Get accuracy badge
badge = tracker.generate_accuracy_badge()
print(f"Badge: {badge['accuracy']} - {badge['status']}")

# Get comparison data for charts
df = tracker.get_prediction_vs_actual_data(days_back=7)
print(df.head())
```

---

## 🚀 **Run Dashboard**

```bash
# Basic run
streamlit run app.py

# Run on specific port
streamlit run app.py --server.port=8502

# Run with custom config
streamlit run app.py --server.address=0.0.0.0 --server.port=8501

# Run with auto-reload disabled (production)
streamlit run app.py --server.runOnSave=false

# Access dashboard
# Local: http://localhost:8501
# Network: http://YOUR_IP:8501
```

---

## 🔄 **Automation Scripts**

### **Daily Data Update**
```python
# scripts/daily_update.py
from dataloader import fetch_openaq_data, fetch_weather_data
import pandas as pd
from datetime import datetime
import os

# Fetch latest data
api_key = os.getenv('OPENAQ_API_KEY')
df_new = fetch_openaq_data(api_key)

# Append to existing
df_existing = pd.read_csv('data/historical_aqi_dataset.csv')
df_combined = pd.concat([df_existing, df_new]).drop_duplicates()
df_combined.to_csv('data/historical_aqi_dataset.csv', index=False)

print(f"Updated {len(df_new)} new records at {datetime.now()}")
```

Run daily with cron:
```bash
# Edit crontab
crontab -e

# Add line (runs at 6 AM daily)
0 6 * * * cd /path/to/project && /path/to/python scripts/daily_update.py
```

### **Weekly Model Retrain**
```python
# scripts/weekly_retrain.py
from ml_models.lstm_forecaster import AQIForecaster
from model_tracker import ModelTracker
import pandas as pd

# Check if retraining needed
tracker = ModelTracker()
accuracy = tracker.get_prediction_accuracy(days_back=7)

if accuracy.get('accuracy_percentage', 100) < 70:
    print("Accuracy below threshold, retraining...")
    
    # Load data
    df = pd.read_csv('data/historical_aqi_dataset.csv')
    df_pm25 = df[df['parameter'] == 'pm25']
    
    # Retrain
    forecaster = AQIForecaster()
    history, metrics = forecaster.train(df_pm25)
    
    print(f"Retraining complete. New MAE: {metrics['mae']:.2f}")
else:
    print(f"Model performing well ({accuracy['accuracy_percentage']:.1f}%), no retraining needed")
```

Run weekly with cron:
```bash
# Sunday at 2 AM
0 2 * * 0 cd /path/to/project && /path/to/python scripts/weekly_retrain.py
```

---

## 🐛 **Debugging**

### **Check Data Quality**
```python
import pandas as pd

df = pd.read_csv('data/historical_aqi_dataset.csv')

print("Data Quality Report:")
print(f"  Total rows: {len(df)}")
print(f"  Missing values:\n{df.isnull().sum()}")
print(f"  Duplicate rows: {df.duplicated().sum()}")
print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"  Locations: {df['location_name'].nunique()}")
print(f"  Parameters: {df['parameter'].unique()}")
```

### **Test Model Prediction**
```python
from ml_models.lstm_forecaster import AQIForecaster
import pandas as pd
import numpy as np

# Create minimal test data
dates = pd.date_range(end='2025-11-02', periods=200, freq='H')
test_data = pd.DataFrame({
    'timestamp': dates,
    'location_name': ['Test'] * len(dates),
    'parameter': ['pm25'] * len(dates),
    'value': 100 + np.random.normal(0, 20, len(dates)),
    'latitude': [28.6] * len(dates),
    'longitude': [77.2] * len(dates),
    'temperature_c': [25] * len(dates),
    'humidity_percent': [60] * len(dates),
    'wind_speed_mps': [3] * len(dates),
    'pressure_hpa': [1013] * len(dates),
    'hour': dates.hour,
    'day_of_week': dates.dayofweek,
    'month': dates.month
})

# Try prediction
try:
    forecaster = AQIForecaster()
    forecaster.load_model('latest')
    forecast = forecaster.predict(test_data, steps_ahead=24)
    print(f"✅ Model working! Forecast: {forecast[:5]}")
except Exception as e:
    print(f"❌ Error: {str(e)}")
```

### **Check API Connectivity**
```python
from dataloader import fetch_openaq_data, fetch_weather_data
import os

# Test OpenAQ
try:
    api_key = os.getenv('OPENAQ_API_KEY')
    df = fetch_openaq_data(api_key)
    print(f"✅ OpenAQ: {len(df)} records")
except Exception as e:
    print(f"❌ OpenAQ error: {str(e)}")

# Test Weather
try:
    api_key = os.getenv('OPENWEATHERMAP_API_KEY')
    weather = fetch_weather_data(api_key)
    print(f"✅ Weather: {weather}")
except Exception as e:
    print(f"❌ Weather error: {str(e)}")
```

---

## 📦 **Deployment**

### **Docker**
```bash
# Build image
docker build -t delhi-aqi-dashboard .

# Run container
docker run -d \
  -p 8501:8501 \
  -e OPENAQ_API_KEY="your_key" \
  -e OPENWEATHERMAP_API_KEY="your_key" \
  --name aqi-dashboard \
  delhi-aqi-dashboard

# View logs
docker logs -f aqi-dashboard

# Stop container
docker stop aqi-dashboard
```

### **Streamlit Cloud**
```bash
# 1. Push to GitHub
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/yourusername/delhi-aqi.git
git push -u origin main

# 2. Go to https://streamlit.io/cloud
# 3. Connect GitHub repo
# 4. Add secrets in dashboard settings
# 5. Deploy!
```

---

## 🧪 **Testing**

```python
# Test geocoding
from geocoding import GeocodingService
assert GeocodingService().search_location("110001") is not None

# Test health advisor
from health_advisor import HealthAdvisor
advisory = HealthAdvisor().get_advisory(150)
assert advisory['category'] in ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe']

# Test model tracker
from model_tracker import ModelTracker
tracker = ModelTracker()
badge = tracker.generate_accuracy_badge()
assert badge['label'] == 'Model Accuracy (7 days)'

print("✅ All tests passed!")
```

---

## 📊 **Performance Monitoring**

```bash
# Monitor Streamlit app
streamlit run app.py --server.enableCORS=false --server.enableXsrfProtection=false

# Check memory usage
python -c "
import psutil
import os
process = psutil.Process(os.getpid())
print(f'Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB')
"

# Profile code
python -m cProfile -o profile.stats ml_models/lstm_forecaster.py
python -c "
import pstats
p = pstats.Stats('profile.stats')
p.sort_stats('cumulative').print_stats(10)
"
```

---

## 🔗 **Useful Links**

- **OpenAQ API:** https://docs.openaq.org
- **CPCB Standards:** https://cpcb.nic.in/air-quality-standard/
- **Streamlit Docs:** https://docs.streamlit.io
- **TensorFlow Tutorials:** https://www.tensorflow.org/tutorials
- **MLflow Tracking:** https://mlflow.org/docs/latest/tracking.html

---

## 💡 **Pro Tips**

1. **Use caching:** Decorate expensive functions with `@st.cache_data`
2. **Version models:** Always save with timestamps
3. **Log everything:** Use Python's `logging` module
4. **Monitor accuracy:** Check model performance weekly
5. **Backup data:** Keep copies of trained models and datasets
6. **Test alerts:** Send test notifications before going live
7. **Mobile-first:** Test on mobile devices
8. **Privacy matters:** Never log personal data
9. **Document changes:** Update README when adding features
10. **Community feedback:** Listen to users!

---

**Happy Coding! 🚀**
