# 🚀 Implementation Guide - Step by Step

## For Data Science & Development Teams

This guide walks through implementing the advanced Delhi AQI dashboard from scratch.

---

## 📋 **Prerequisites**

### **Skills Required**
- Python 3.11+
- Basic understanding of:
  - Pandas/NumPy for data processing
  - Machine learning concepts (LSTM basics helpful but not required)
  - Streamlit for web dashboards
  - REST APIs
  - Git/GitHub

### **Accounts & API Keys Needed**
1. **OpenAQ** - Free account at https://openaq.org
2. **OpenWeatherMap** - Free tier at https://openweathermap.org
3. **SendGrid** (Optional) - Free tier at https://sendgrid.com (12,000 emails/month)
4. **Twilio** (Optional) - Trial account at https://twilio.com
5. **GitHub** - For version control and deployment

---

## 🏗️ **Phase 1: Project Setup** (Day 1)

### **Step 1.1: Environment Setup**

```bash
# Create project directory
mkdir delhi_aqi_dashboard
cd delhi_aqi_dashboard

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or use Conda
conda create -n delhi_aqi python=3.11
conda activate delhi_aqi
```

### **Step 1.2: Install Dependencies**

```bash
# Copy requirements.txt from the project
pip install -r requirements.txt

# Verify installation
python -c "import streamlit; import tensorflow; import geopandas; print('✅ All packages installed')"
```

### **Step 1.3: Configure API Keys**

Create `.streamlit/secrets.toml`:
```toml
[api_keys]
openaq_key = "YOUR_OPENAQ_API_KEY"
openweathermap_key = "YOUR_OWM_API_KEY"
sendgrid_api_key = "YOUR_SENDGRID_KEY"  # Optional

[twilio]  # Optional
account_sid = "YOUR_TWILIO_SID"
auth_token = "YOUR_TWILIO_TOKEN"
from_number = "+1234567890"
```

Or use environment variables:
```bash
export OPENAQ_API_KEY="your_key_here"
export OPENWEATHERMAP_API_KEY="your_key_here"
```

### **Step 1.4: Download Geospatial Data**

Your `data/` folder already has:
- `delhi_districts.json`
- `delhi_assembly-constituencie.geojson`

If you need more detailed boundaries:
- Delhi Municipal Corporation Wards: https://data.gov.in
- MMRDA boundaries: https://portal.spatial.dcgis.delhi.gov.in

---

## 📊 **Phase 2: Data Collection** (Days 2-3)

### **Step 2.1: Test API Connections**

```python
# Test OpenAQ connection
python -c "
from dataloader import fetch_openaq_data
import os
api_key = os.getenv('OPENAQ_API_KEY')
df = fetch_openaq_data(api_key)
print(f'Fetched {len(df)} records')
print(df.head())
"

# Test Weather API
python -c "
from dataloader import fetch_weather_data
import os
api_key = os.getenv('OPENWEATHERMAP_API_KEY')
weather = fetch_weather_data(api_key)
print(weather)
"
```

### **Step 2.2: Run Historical Data Pipeline**

**Important:** This will take 2-4 hours depending on API rate limits.

```bash
# Fetch 10 years of data
python data_pipeline.py

# Expected output:
# - data/historical_aqi_dataset.csv (~50-100 MB)
# - data/dataset_metadata.json
```

**Troubleshooting:**
- If API rate limits are hit, the pipeline will use sample data
- You can adjust `years_back` parameter in `HistoricalDataPipeline(years_back=2)` for faster testing

### **Step 2.3: Verify Data Quality**

```python
import pandas as pd

df = pd.read_csv('data/historical_aqi_dataset.csv')

print(f"Dataset shape: {df.shape}")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"Locations: {df['location_name'].nunique()}")
print(f"Parameters: {df['parameter'].unique()}")
print(f"Missing values:\n{df.isnull().sum()}")
```

Expected output:
- 500K - 2M rows (depending on years and locations)
- At least 5-10 unique locations
- pm25, pm10, no2, so2, o3 parameters
- < 5% missing values (after forward/backward fill)

---

## 🤖 **Phase 3: Model Training** (Days 4-5)

### **Step 3.1: Prepare Training Data**

```bash
# Filter for PM2.5 (primary pollutant)
python -c "
import pandas as pd
df = pd.read_csv('data/historical_aqi_dataset.csv')
df_pm25 = df[df['parameter'] == 'pm25'].copy()
df_pm25.to_csv('data/training_data_pm25.csv', index=False)
print(f'PM2.5 records: {len(df_pm25)}')
"
```

### **Step 3.2: Train LSTM Model**

```bash
# Initial training (may take 1-3 hours on CPU, 10-30 min on GPU)
python ml_models/lstm_forecaster.py

# Monitor training in terminal
# You should see:
# - Epoch progress
# - Loss decreasing
# - Validation metrics
# - Model saved to models/versions/YYYYMMDD_HHMMSS/
```

**Training Tips:**
- **CPU users:** Reduce batch size to 16, epochs to 20 for faster training
- **GPU users:** Can use larger batch sizes (64-128)
- **Small datasets:** Use sequence_length=72 (3 days) instead of 168

### **Step 3.3: Evaluate Model**

```python
from ml_models.lstm_forecaster import AQIForecaster

forecaster = AQIForecaster()
metadata = forecaster.load_model(version='latest')

print("Model Info:")
print(f"  Version: {metadata['version']}")
print(f"  Created: {metadata['created_at']}")
print(f"  Features: {len(metadata['feature_columns'])}")

# Test prediction
import pandas as pd
recent_data = pd.read_csv('data/training_data_pm25.csv').tail(200)
forecast = forecaster.predict(recent_data, steps_ahead=24)

print(f"\nNext 24-hour forecast: {forecast}")
```

Expected model performance:
- **MAE:** 15-30 µg/m³ (lower is better)
- **RMSE:** 20-40 µg/m³
- **R²:** 0.65-0.85 (higher is better)

---

## 🌐 **Phase 4: Dashboard Development** (Days 6-7)

### **Step 4.1: Test Basic Dashboard**

```bash
# Run existing app.py
streamlit run app.py

# Should open at http://localhost:8501
# Verify:
# - Map loads with stations
# - Basic charts display
# - No errors in terminal
```

### **Step 4.2: Integrate Advanced Features**

Copy sections from `app_enhanced_example.py` into your `app.py`:

**Priority Order:**
1. **Location search** (geocoding.py)
2. **Forecasting tab** (ml_models/lstm_forecaster.py)
3. **Model performance** (model_tracker.py)
4. **Health advisories** (health_advisor.py)
5. **Alert subscriptions** (notifications.py)

**Integration Example:**
```python
# Add to top of app.py
from geocoding import GeocodingService
from ml_models.lstm_forecaster import AQIForecaster
from model_tracker import ModelTracker
from health_advisor import HealthAdvisor

# Initialize in main()
@st.cache_resource
def init_services():
    return {
        'geocoder': GeocodingService(),
        'forecaster': AQIForecaster(),
        'tracker': ModelTracker(),
        'advisor': HealthAdvisor()
    }

services = init_services()
```

### **Step 4.3: Test Each Feature**

1. **Location Search:**
   ```
   - Try: "Connaught Place"
   - Try: "110001"
   - Verify ward/district display
   ```

2. **Forecasting:**
   ```
   - Load model ✓
   - Generate 24h forecast ✓
   - Display chart ✓
   - Show health advisory ✓
   ```

3. **Model Tracking:**
   ```
   - Display accuracy badge ✓
   - Show predictions vs actuals ✓
   - Calculate metrics ✓
   ```

---

## 🔔 **Phase 5: Notifications** (Day 8)

### **Step 5.1: Configure SendGrid (Email)**

1. Sign up at https://sendgrid.com
2. Create API key (Settings → API Keys)
3. Verify sender email
4. Add to secrets.toml

Test:
```python
from notifications import NotificationService

service = NotificationService()
service.send_email_alert(
    to_email="your.email@example.com",
    subject="Test Alert",
    content="This is a test email from Delhi AQI Dashboard"
)
```

### **Step 5.2: Configure Twilio (SMS - Optional)**

1. Sign up at https://twilio.com
2. Get phone number
3. Get Account SID and Auth Token
4. Add to secrets.toml

Test:
```python
service.send_sms_alert(
    to_phone="+919876543210",  # Your phone with country code
    message="Test SMS from Delhi AQI Dashboard"
)
```

### **Step 5.3: Set Up Alert Automation**

Create `scripts/send_daily_forecasts.py`:
```python
from notifications import NotificationService
from ml_models.lstm_forecaster import AQIForecaster

service = NotificationService()
forecaster = AQIForecaster()
forecaster.load_model('latest')

# Get forecast for each subscribed location
# If AQI > threshold, send alert
# (Implementation in app_enhanced_example.py)
```

Run daily via cron:
```bash
0 6 * * * cd /path/to/project && python scripts/send_daily_forecasts.py
```

---

## 🔄 **Phase 6: Automation** (Days 9-10)

### **Step 6.1: Daily Data Update Script**

Create `scripts/update_data.py`:
```python
from dataloader import fetch_openaq_data, fetch_weather_data
import pandas as pd
from datetime import datetime

# Fetch last 24 hours
df_new = fetch_openaq_data(api_key)

# Append to historical dataset
df_existing = pd.read_csv('data/historical_aqi_dataset.csv')
df_combined = pd.concat([df_existing, df_new]).drop_duplicates()
df_combined.to_csv('data/historical_aqi_dataset.csv', index=False)

print(f"Updated at {datetime.now()}")
```

### **Step 6.2: Weekly Model Retraining**

Create `scripts/retrain_model.py`:
```python
from ml_models.lstm_forecaster import AQIForecaster
from model_tracker import ModelTracker

tracker = ModelTracker()
current_accuracy = tracker.get_prediction_accuracy(days_back=7)

# Retrain if accuracy < 70%
if current_accuracy.get('accuracy_percentage', 100) < 70:
    print("Retraining model...")
    forecaster = AQIForecaster()
    # Train on updated dataset
    # ...
```

### **Step 6.3: Airflow DAGs (Production)**

If using Airflow for production:

```python
# dags/aqi_pipeline.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'aqi_team',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'aqi_daily_pipeline',
    default_args=default_args,
    schedule_interval='0 6 * * *'  # 6 AM daily
)

update_data = PythonOperator(
    task_id='update_data',
    python_callable=run_data_update,
    dag=dag
)

generate_forecasts = PythonOperator(
    task_id='generate_forecasts',
    python_callable=run_forecasts,
    dag=dag
)

send_alerts = PythonOperator(
    task_id='send_alerts',
    python_callable=run_alerts,
    dag=dag
)

update_data >> generate_forecasts >> send_alerts
```

---

## 🚢 **Phase 7: Deployment** (Days 11-12)

### **Option A: Streamlit Cloud (Easiest)**

1. Push code to GitHub
2. Go to https://streamlit.io/cloud
3. Connect GitHub repository
4. Add secrets in dashboard settings
5. Deploy!

**Pros:** Free, easy, automatic HTTPS
**Cons:** Limited compute, public by default

### **Option B: Docker + AWS/GCP**

Create `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t delhi-aqi-dashboard .
docker run -p 8501:8501 --env-file .env delhi-aqi-dashboard
```

Deploy to AWS ECS, Google Cloud Run, or Azure Container Instances.

### **Option C: VPS (DigitalOcean, Linode, etc.)**

```bash
# Install dependencies
sudo apt update
sudo apt install python3.11 python3-pip nginx

# Clone repo
git clone https://github.com/yourusername/delhi-aqi-dashboard
cd delhi-aqi-dashboard

# Install packages
pip install -r requirements.txt

# Run with PM2 or systemd
pm2 start "streamlit run app.py" --name aqi-dashboard

# Configure Nginx reverse proxy
# ...
```

---

## 📊 **Phase 8: Monitoring & Maintenance**

### **8.1: Set Up Monitoring**

**Application Monitoring:**
- **Streamlit Cloud:** Built-in analytics
- **Self-hosted:** Use Prometheus + Grafana

**Model Performance Monitoring:**
```python
# Track daily in model_tracker.py
tracker = ModelTracker()
daily_metrics = tracker.get_prediction_accuracy(days_back=1)

# Alert if accuracy drops
if daily_metrics['accuracy_percentage'] < 60:
    send_admin_alert("Model accuracy degraded!")
```

**Data Quality Checks:**
```python
# scripts/data_quality_check.py
import pandas as pd

df = pd.read_csv('data/historical_aqi_dataset.csv')

# Check for gaps
df['timestamp'] = pd.to_datetime(df['timestamp'])
gaps = df.groupby('location_name')['timestamp'].apply(
    lambda x: (x.diff() > pd.Timedelta('2 hours')).sum()
)

if gaps.sum() > 10:
    send_admin_alert(f"Data gaps detected: {gaps}")
```

### **8.2: Regular Maintenance Tasks**

**Daily:**
- ✓ Fetch new AQI data
- ✓ Generate forecasts
- ✓ Send scheduled alerts
- ✓ Check API status

**Weekly:**
- ✓ Evaluate model accuracy
- ✓ Retrain if needed
- ✓ Clean old logs
- ✓ Backup database

**Monthly:**
- ✓ Full model retraining
- ✓ Update event markers
- ✓ Review user feedback
- ✓ Update geospatial data

---

## 🐛 **Common Issues & Solutions**

### **Issue 1: Model Won't Load**
```
FileNotFoundError: No saved models found
```
**Solution:** Run training first: `python ml_models/lstm_forecaster.py`

### **Issue 2: API Rate Limits**
```
OpenAQ API: 429 Too Many Requests
```
**Solution:** 
- Add delays between requests
- Cache data locally
- Upgrade to paid tier if needed

### **Issue 3: GeoPandas Installation Fails**
```
ERROR: Failed building wheel for geopandas
```
**Solution:**
```bash
# On Ubuntu/Debian
sudo apt-get install gdal-bin libgdal-dev

# On macOS
brew install gdal

# Then reinstall
pip install geopandas
```

### **Issue 4: TensorFlow GPU Not Detected**
```
Solution:
# Install CUDA and cuDNN
# See: https://www.tensorflow.org/install/gpu

# Verify
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### **Issue 5: Streamlit App Slow**
**Solutions:**
- Use `@st.cache_data` and `@st.cache_resource` decorators
- Reduce data loading frequency
- Paginate large tables
- Optimize map rendering

---

## 📚 **Learning Resources**

### **LSTM & Time Series**
- [TensorFlow Time Series Tutorial](https://www.tensorflow.org/tutorials/structured_data/time_series)
- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

### **Streamlit**
- [Official Documentation](https://docs.streamlit.io)
- [Gallery of Apps](https://streamlit.io/gallery)

### **Air Quality**
- [OpenAQ Documentation](https://docs.openaq.org)
- [CPCB AQI Standards](https://cpcb.nic.in/air-quality-standard/)

---

## 🎯 **Success Metrics**

Track these KPIs:

1. **User Engagement:**
   - Daily active users
   - Average session duration
   - Locations searched per day

2. **Model Performance:**
   - Forecast accuracy (target: >75%)
   - Mean absolute error (target: <25 µg/m³)
   - R² score (target: >0.70)

3. **Citizen Impact:**
   - Alert subscribers
   - Alert open rate
   - Social media shares
   - User feedback score

---

## 📞 **Getting Help**

- **GitHub Issues:** Report bugs or request features
- **Discussion Forum:** Ask questions, share ideas
- **Email:** support@delhiaqi.app (if deployed)

---

## ✅ **Deployment Checklist**

Before going live:

- [ ] All API keys configured
- [ ] Model trained and validated
- [ ] Dashboard tested on multiple browsers
- [ ] Responsive design verified (mobile/tablet)
- [ ] Error handling implemented
- [ ] Monitoring set up
- [ ] Backup strategy in place
- [ ] Privacy policy added
- [ ] Terms of service added
- [ ] Contact information displayed
- [ ] Social media accounts created
- [ ] Documentation complete
- [ ] Team trained on maintenance

---

**Good luck building a dashboard that helps citizens breathe easier! 🌍💚**
