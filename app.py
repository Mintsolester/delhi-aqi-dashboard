"""
AI-Powered Spatio-Temporal Dashboard for Delhi Environmental Health
Main Streamlit Application with LSTM Forecasting - MEMORY OPTIMIZED
"""

import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path
import sys
import gc
from sklearn.exceptions import NotFittedError

# Add ml_models to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'ml_models'))

from dataloader import fetch_openaq_data, fetch_weather_data, load_geospatial_boundaries
from data_processor import harmonize_aq_weather, perform_spatial_join, aggregate_by_geography
from analytics import prepare_features, train_regression_model, generate_insights

# Import advanced modules
try:
    from geocoding import GeocodingService
    from health_advisor import HealthAdvisor
    from notifications import NotificationService
    from ml_models.lstm_forecaster import AQIForecaster
    from config import config
    ADVANCED_FEATURES_AVAILABLE = True
except ImportError as e:
    ADVANCED_FEATURES_AVAILABLE = False
    st.sidebar.warning(f"Some advanced features unavailable: {e}")

# Page configuration
st.set_page_config(
    page_title="Delhi Environmental Health Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 20px 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        padding-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)

def get_aqi_category(value, parameter='pm25'):
    """Determine AQI category"""
    if parameter.lower() in ['pm25', 'pm2.5']:
        if value <= 30: return 'Good', '#00e400'
        elif value <= 60: return 'Satisfactory', '#ffff00'
        elif value <= 90: return 'Moderate', '#ff7e00'
        elif value <= 120: return 'Poor', '#ff0000'
        elif value <= 250: return 'Very Poor', '#8f3f97'
        else: return 'Severe', '#7e0023'
    else:
        if value <= 50: return 'Good', '#00e400'
        elif value <= 100: return 'Moderate', '#ffff00'
        elif value <= 200: return 'Poor', '#ff7e00'
        else: return 'Very Poor', '#ff0000'

@st.cache_resource
def init_services():
    """Initialize advanced services - LAZY LOADING"""
    if not ADVANCED_FEATURES_AVAILABLE:
        return None, None, None, None
    
    try:
        geocoding = GeocodingService()
        health_advisor = HealthAdvisor()
        notifier = NotificationService()
        
        # Don't load LSTM model here - load it only when needed
        forecaster = None
        
        return geocoding, health_advisor, notifier, forecaster
    except Exception as e:
        st.sidebar.error(f"Service init error: {e}")
        return None, None, None, None

@st.cache_resource
def load_lstm_model():
    """Lazy load LSTM model only when forecast tab is accessed"""
    try:
        import importlib
        import importlib.util
        load_model = None
        # Try to dynamically load TensorFlow's load_model; fallback to standalone Keras if TF not available
        tf_spec = importlib.util.find_spec("tensorflow")
        if tf_spec:
            tf = importlib.import_module("tensorflow")
            # Prefer tf.keras.models.load_model when available
            load_model = getattr(tf.keras.models, "load_model", None)
        else:
            keras_spec = importlib.util.find_spec("keras")
            if keras_spec:
                keras = importlib.import_module("keras")
                load_model = getattr(keras.models, "load_model", None)
        import joblib
        import json
        if load_model is None:
            raise ImportError("Keras 'load_model' function not found; install tensorflow or keras")

        # Configure TensorFlow for memory efficiency
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                st.warning(f"GPU config: {e}")
        
        # Try .keras first (recommended), then .h5 (legacy)
        model_path_keras = config.MODEL_DIR / 'best_model.keras'
        model_path_h5 = config.MODEL_DIR / 'best_model.h5'
        
        # DEBUG: Print paths being checked
        st.sidebar.info(f"🔍 Looking for models in: {config.MODEL_DIR}")
        st.sidebar.info(f"📂 Dir exists: {config.MODEL_DIR.exists()}")
        if config.MODEL_DIR.exists():
            st.sidebar.info(f"📄 Files: {list(config.MODEL_DIR.glob('*'))}")
        
        model_path = model_path_keras if model_path_keras.exists() else model_path_h5
        
        scaler_X_path = config.MODEL_DIR / 'scaler_X.pkl'
        scaler_y_path = config.MODEL_DIR / 'scaler_y.pkl'
        feature_cols_path = config.MODEL_DIR / 'feature_columns.json'
        
        if model_path.exists() and scaler_X_path.exists() and scaler_y_path.exists():
            forecaster = AQIForecaster(sequence_length=72, forecast_horizon=24)
            
            # Load model
            forecaster.model = load_model(str(model_path))
            
            # Load scalers
            forecaster.scaler_X = joblib.load(scaler_X_path)
            forecaster.scaler_y = joblib.load(scaler_y_path)
            
            # Load feature columns
            if feature_cols_path.exists():
                with open(feature_cols_path, 'r') as f:
                    metadata = json.load(f)
                    forecaster.feature_columns = metadata['feature_columns']
                    forecaster.sequence_length = metadata['sequence_length']
                    forecaster.forecast_horizon = metadata['forecast_horizon']
            
            format_type = "native Keras" if model_path.suffix == '.keras' else "HDF5 (legacy)"
            st.sidebar.success(f"✅ LSTM model loaded ({format_type})")
            return forecaster
        else:
            missing = []
            if not model_path.exists():
                missing.append("best_model.keras/.h5")
            if not scaler_X_path.exists():
                missing.append("scaler_X.pkl")
            if not scaler_y_path.exists():
                missing.append("scaler_y.pkl")
            
            st.sidebar.warning(f"❌ Missing: {', '.join(missing)}")
            st.sidebar.info("ℹ️ Train model: python ml_models/lstm_forecaster.py")
            return None
        
    except Exception as e:
        st.sidebar.error(f"Model load error: {str(e)}")
        return None

@st.cache_data(ttl=1800, max_entries=1)
def load_forecast_data():
    """Load ONLY the data needed for LSTM forecasting - called only when forecast tab is accessed"""
    try:
        hist_data_path = config.DATA_DIR / "historical_aqi_dataset.csv"
        
        if not hist_data_path.exists():
            return None
        
        # Load ONLY columns needed for forecasting (17 base features + value/timestamp/parameter)
        forecast_cols = [
            'parameter', 'value', 'timestamp',
            # Weather features (7 features)
            'temperature_c', 'humidity_percent', 'wind_speed_mps', 
            'pressure_hpa', 'cloud_cover', 'precipitation_mm', 'visibility_km',
            # Temporal features (10 features)
            'hour', 'day_of_week', 'month', 'is_weekend', 
            'is_crop_burning_season', 'is_winter', 'is_diwali', 'is_holiday'
        ]
        
        # Load only last 1000 rows (enough for 72h sequence + lagging + rolling)
        df_hist = pd.read_csv(
            hist_data_path,
            usecols=forecast_cols,
            dtype={
                'parameter': 'category',
                'value': 'float32',
                'temperature_c': 'float32',
                'humidity_percent': 'float32',
                'wind_speed_mps': 'float32',
                'pressure_hpa': 'float32',
                'cloud_cover': 'float32',
                'precipitation_mm': 'float32',
                'visibility_km': 'float32',
                'hour': 'int8',
                'day_of_week': 'int8',
                'month': 'int8',
                'is_weekend': 'bool',
                'is_crop_burning_season': 'bool',
                'is_winter': 'bool',
                'is_diwali': 'bool',
                'is_holiday': 'bool'
            }
        )
        
        # Get last 500 PM2.5 records (enough for feature engineering + sequences)
        df_pm25 = df_hist[df_hist['parameter'] == 'pm25'].tail(500).copy()
        
        # Clean up
        del df_hist
        gc.collect()
        
        return df_pm25
        
    except Exception as e:
        st.error(f"Error loading forecast data: {e}")
        return None

@st.cache_data(ttl=3600, max_entries=1)
def load_all_data():
    """Load and process all data with aggressive memory optimization"""
    try:
        openaq_key = st.secrets.get("api_keys", {}).get("openaq_key", "")
        owm_key = st.secrets.get("api_keys", {}).get("openweathermap_key", "")
        
        aq_df = fetch_openaq_data(openaq_key)
        weather_dict = fetch_weather_data(owm_key)
        districts_gdf, wards_gdf = load_geospatial_boundaries()
        
        integrated_df = harmonize_aq_weather(aq_df, weather_dict)
        integrated_gdf = perform_spatial_join(integrated_df, wards_gdf)
        
        # Memory optimization: keep only essential columns
        essential_cols = ['location_name', 'latitude', 'longitude', 'parameter', 
                         'value', 'unit', 'timestamp', 'ward', 'district']
        if 'geometry' in integrated_gdf.columns:
            essential_cols.append('geometry')
        
        integrated_gdf = integrated_gdf[[col for col in essential_cols if col in integrated_gdf.columns]]
        
        # Convert to more memory-efficient dtypes
        for col in ['value', 'latitude', 'longitude']:
            if col in integrated_gdf.columns:
                integrated_gdf[col] = integrated_gdf[col].astype('float32')
        
        for col in ['location_name', 'parameter', 'unit', 'ward', 'district']:
            if col in integrated_gdf.columns:
                integrated_gdf[col] = integrated_gdf[col].astype('category')
        
        # Simplify geometries if present
        if 'geometry' in districts_gdf.columns:
            districts_gdf['geometry'] = districts_gdf['geometry'].simplify(0.001)
        if 'geometry' in wards_gdf.columns:
            wards_gdf['geometry'] = wards_gdf['geometry'].simplify(0.001)
        
        # Force garbage collection
        gc.collect()
        
        return integrated_gdf, districts_gdf, wards_gdf, weather_dict
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None, None

def main():
    st.markdown('<div class="main-header">🌍 Delhi Environmental Health Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered LSTM Forecasting & Spatio-Temporal Monitoring</div>', unsafe_allow_html=True)
    
    # Initialize services (without LSTM model)
    geocoding, health_advisor, notifier, _ = init_services()
    
    with st.spinner("Loading data from APIs..."):
        integrated_gdf, districts_gdf, wards_gdf, weather_dict = load_all_data()
    
    if integrated_gdf is None:
        st.error("Failed to load data. Check API keys and internet connection.")
        st.stop()
    
    # Sidebar
    st.sidebar.title("📊 Dashboard Controls")
    st.sidebar.markdown("---")
    
    current_time = datetime.now()
    st.sidebar.info(f"**Last Updated:** {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    available_parameters = integrated_gdf['parameter'].unique().tolist()
    selected_parameter = st.sidebar.selectbox(
        "Select Pollutant",
        options=available_parameters,
        index=0 if 'pm25' in available_parameters else 0
    )
    
    geography_level = st.sidebar.radio(
        "Geography Level",
        options=['Ward', 'District'],
        index=0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🌤️ Current Weather")
    if weather_dict:
        st.sidebar.metric("Temperature", f"{weather_dict.get('temperature_c', 'N/A')}°C")
        st.sidebar.metric("Humidity", f"{weather_dict.get('humidity_percent', 'N/A')}%")
        st.sidebar.metric("Wind Speed", f"{weather_dict.get('wind_speed_mps', 'N/A')} m/s")
    
    # Tabs - Added new tabs for advanced features
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "🗺️ Live Air Quality Map",
        "🔮 LSTM Forecast (24h)",
        "🔍 Location Search",
        "💊 Health Advisory",
        "📈 Historical Trends",
        "🤖 AI Analytics",
        "🔔 Alerts & Notifications",
        "📋 Data Table"
    ])
    
    # Tab 1: Map
    with tab1:
        st.subheader(f"Spatial Distribution of {selected_parameter.upper()}")
        
        param_data = integrated_gdf[integrated_gdf['parameter'] == selected_parameter].copy()
        
        if len(param_data) == 0:
            st.warning(f"No data available for {selected_parameter}")
        else:
            # Limit markers to prevent memory issues
            max_markers = 50
            if len(param_data) > max_markers:
                st.info(f"Showing {max_markers} of {len(param_data)} stations")
                param_data = param_data.head(max_markers)
            
            m = folium.Map(location=[28.6139, 77.2090], zoom_start=11, tiles='CartoDB positron')
            
            for idx, row in param_data.iterrows():
                category, color = get_aqi_category(row['value'], selected_parameter)
                
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=8,
                    popup=f"""
                    <b>{row['location_name']}</b><br>
                    {selected_parameter.upper()}: {row['value']:.2f} {row['unit']}<br>
                    Category: {category}<br>
                    Updated: {row['timestamp']}
                    """,
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.7,
                    weight=2
                ).add_to(m)
            
            st_folium(m, width=1200, height=600)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Stations", len(param_data))
            with col2:
                st.metric("Average", f"{param_data['value'].mean():.2f}")
            with col3:
                st.metric("Max", f"{param_data['value'].max():.2f}")
            with col4:
                st.metric("Min", f"{param_data['value'].min():.2f}")
            
            # Clean up
            del param_data
            gc.collect()
    
    # Tab 2: LSTM Forecast - LAZY LOAD MODEL AND DATA
    with tab2:
        st.subheader("🔮 24-Hour AQI Forecast (LSTM Model)")
        
        # Load model only when this tab is accessed
        forecaster = load_lstm_model()
        
        if forecaster and forecaster.model:
            # Load forecast data only when needed (separate cache from main data)
            with st.spinner("Loading historical data for forecasting..."):
                df_pm25 = load_forecast_data()
            
            if df_pm25 is not None and len(df_pm25) >= 120:
                try:
                    # Make prediction with error handling
                    with st.spinner("Generating 24-hour forecast..."):
                        predictions = forecaster.predict(df_pm25, steps_ahead=24)
                    
                    # Create forecast visualization
                    forecast_hours = [(datetime.now() + timedelta(hours=i)).strftime('%H:%00') 
                                     for i in range(1, 25)]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=forecast_hours,
                        y=predictions,
                        mode='lines+markers',
                        name='Predicted PM2.5',
                        line=dict(color='#ff7e00', width=3),
                        marker=dict(size=8)
                    ))
                    
                    # Add AQI category thresholds
                    fig.add_hline(y=30, line_dash="dash", line_color="green", 
                                 annotation_text="Good/Satisfactory")
                    fig.add_hline(y=60, line_dash="dash", line_color="yellow",
                                 annotation_text="Moderate")
                    fig.add_hline(y=90, line_dash="dash", line_color="orange",
                                 annotation_text="Poor")
                    fig.add_hline(y=120, line_dash="dash", line_color="red",
                                 annotation_text="Very Poor")
                    
                    fig.update_layout(
                        title="Next 24-Hour PM2.5 Forecast",
                        xaxis_title="Hour",
                        yaxis_title="PM2.5 (µg/m³)",
                        height=500,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, width='stretch')
                    
                    # Show forecast summary
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Avg Forecast", f"{predictions.mean():.1f} µg/m³")
                    with col2:
                        st.metric("Max Expected", f"{predictions.max():.1f} µg/m³")
                    with col3:
                        peak_hour = forecast_hours[np.argmax(predictions)]
                        st.metric("Peak Hour", peak_hour)
                    
                    # Health advisory based on forecast
                    if health_advisor:
                        max_forecast = predictions.max()
                        st.subheader("Health Advisory for Tomorrow")
                        advisory = health_advisor.get_advisory(
                            aqi_value=max_forecast,
                            vulnerable_groups=None
                        )
                        
                        if advisory:
                            category = advisory.get('category', 'Unknown')
                            st.markdown(f"**Category:** :{'red' if max_forecast > 90 else 'orange' if max_forecast > 60 else 'green'}[{category}]")
                            
                            if advisory.get('summary'):
                                st.info(f"**{advisory['icon']} {advisory['summary']}**")
                            
                            recommendations = advisory.get('recommendations', [])
                            if recommendations and len(recommendations) > 0:
                                st.warning(f"**Recommendation:** {recommendations[0]}")
                    
                    # Clean up
                    del predictions
                    gc.collect()
                except NotFittedError:
                    st.error("Forecast error: The model's scaler is not fitted.")
                    st.info("Train the model properly using: `python ml_models/lstm_forecaster.py`")
                except Exception as pred_error:
                    st.error(f"Prediction error: {str(pred_error)}")
                    st.info("Make sure the LSTM model and scalers are trained correctly.")
            elif df_pm25 is None:
                st.warning("Historical data not found. Train model first: `python ml_models/lstm_forecaster.py`")
            else:
                st.warning(f"Insufficient historical data. Need at least 120 records, found {len(df_pm25)}")
                st.info("The model requires more historical data to create lagged/rolling features and sequences.")
        else:
            st.warning("⚠️ LSTM model not loaded")
            st.info("""
            **To enable forecasting:**
            1. Train the model: `python ml_models/lstm_forecaster.py`
            2. Restart the dashboard
            """)
    
    # Tab 3: Location Search
    with tab3:
        st.subheader("🔍 Search AQI by Location")
        
        if geocoding:
            search_type = st.radio("Search by:", ["Address/Pincode", "Coordinates"])
            
            if search_type == "Address/Pincode":
                query = st.text_input("Enter Address or 6-digit Pincode", "Connaught Place, New Delhi")
                if st.button("Search"):
                    with st.spinner("Searching..."):
                        result = geocoding.search_location(query)
                        if result:
                            st.success(f"📍 Found: {result.get('formatted_address', query)}")
                            st.write(f"Coordinates: {result['latitude']:.4f}, {result['longitude']:.4f}")
                            
                            if result.get('ward'):
                                st.write(f"Ward: {result['ward']}")
                            if result.get('district'):
                                st.write(f"District: {result['district']}")
                            
                            # Find nearest monitoring station
                            try:
                                nearest = geocoding.find_nearest_monitoring_station(
                                    result['latitude'],
                                    result['longitude'],
                                    integrated_gdf
                                )
                                if nearest and isinstance(nearest, dict):
                                    location_name = nearest.get('location_name', 'Unknown Station')
                                    distance = nearest.get('distance_km', nearest.get('distance', 0))
                                    value = nearest.get('value', 0)
                                    
                                    st.write(f"Nearest Station: {location_name} ({distance:.2f} km)")
                                    st.metric("Current AQI", f"{value:.1f} µg/m³")
                                else:
                                    st.warning("No monitoring stations found nearby")
                            except Exception as e:
                                st.error(f"Error finding nearest station: {str(e)}")
                        else:
                            st.error("Location not found")
            
            else:
                col1, col2 = st.columns(2)
                with col1:
                    lat = st.number_input("Latitude", value=28.6139, format="%.4f")
                with col2:
                    lon = st.number_input("Longitude", value=77.2090, format="%.4f")
                
                if st.button("Search"):
                    try:
                        nearest = geocoding.find_nearest_monitoring_station(lat, lon, integrated_gdf)
                        if nearest and isinstance(nearest, dict):
                            location_name = nearest.get('location_name', 'Unknown Station')
                            distance = nearest.get('distance_km', nearest.get('distance', 0))
                            value = nearest.get('value', 0)
                            
                            st.success(f"Nearest Station: {location_name}")
                            st.metric("Distance", f"{distance:.2f} km")
                            st.metric("Current AQI", f"{value:.1f} µg/m³")
                        else:
                            st.warning("No monitoring stations found nearby")
                    except Exception as e:
                        st.error(f"Error finding nearest station: {str(e)}")
        else:
            st.warning("Geocoding service not available")
    
    # Tab 4: Health Advisory
    with tab4:
        st.subheader("💊 Personalized Health Advisory")
        
        if health_advisor:
            param_data = integrated_gdf[integrated_gdf['parameter'] == selected_parameter]
            avg_aqi = param_data['value'].mean() if len(param_data) > 0 else 100
            
            vulnerable_group = st.selectbox(
                "Select your profile:",
                ["general", "elderly", "children", "asthmatic", "heart_disease", "pregnant"]
            )
            
            groups_list = [vulnerable_group] if vulnerable_group != 'general' else None
            
            advisory = health_advisor.get_advisory(
                aqi_value=avg_aqi,
                vulnerable_groups=groups_list
            )
            
            if advisory:
                category = advisory.get('category', 'Unknown')
                
                st.markdown(f"### Current Air Quality: :{'red' if avg_aqi > 90 else 'orange' if avg_aqi > 60 else 'green'}[{category}]")
                st.metric("Average AQI", f"{avg_aqi:.1f} µg/m³")
                
                if advisory.get('summary'):
                    st.markdown(f"#### {advisory['icon']} {advisory['summary']}")
                
                st.markdown("#### ⚠️ Health Impacts")
                st.warning(advisory.get('health_impact', 'Monitor air quality regularly'))
                
                st.markdown("#### 📋 Recommendations")
                recommendations = advisory.get('recommendations', [])
                if recommendations:
                    for rec in recommendations:
                        st.info(f"• {rec}")
                else:
                    st.info("Stay informed about air quality")
                
                st.markdown("#### 🏃 Outdoor Activity Guidance")
                st.success(advisory.get('outdoor_activity', 'Adjust outdoor activities based on air quality'))
                
                if advisory.get('mask_needed'):
                    st.markdown(f"#### 😷 Mask Recommendation")
                    st.warning(f"**Recommended:** {advisory.get('mask_type', 'N95 mask')}")
                
                if vulnerable_group != 'general' and advisory.get('vulnerable_groups'):
                    group_advice = advisory['vulnerable_groups'].get(vulnerable_group, {})
                    if group_advice:
                        st.markdown(f"#### {group_advice.get('icon', '⚕️')} Specific Advice for {group_advice.get('name', vulnerable_group)}")
                        st.error(f"**Risk Level:** {group_advice.get('risk_level', 'Monitor')}")
                        
                        specific_advice = group_advice.get('specific_advice', [])
                        if specific_advice:
                            for advice in specific_advice:
                                st.warning(f"• {advice}")
                
                if st.button("📱 Get Shareable Message"):
                    whatsapp_msg = health_advisor.get_whatsapp_message(
                        avg_aqi, category, vulnerable_group
                    )
                    st.code(whatsapp_msg, language=None)
                    st.caption("Copy and share via WhatsApp")
        else:
            st.warning("Health advisory service not available")
    
    # Tab 5: Trends
    with tab5:
        st.subheader("Temporal Analysis")
        st.info("📝 Note: This is a PoC. Historical visualization requires time-series database.")
        
        dates = pd.date_range(end=datetime.now(), periods=24, freq='h')  # Fixed deprecation warning
        locations = integrated_gdf['location_name'].unique()[:5]
        
        fig = go.Figure()
        
        for location in locations:
            current_val = integrated_gdf[
                (integrated_gdf['location_name'] == location) & 
                (integrated_gdf['parameter'] == selected_parameter)
            ]['value'].values
            
            if len(current_val) > 0:
                base_value = current_val[0]
                simulated_values = base_value + np.random.normal(0, base_value * 0.2, len(dates))
                simulated_values = np.clip(simulated_values, 0, None)
                
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=simulated_values,
                    mode='lines+markers',
                    name=location
                ))
        
        fig.update_layout(
            title=f"{selected_parameter.upper()} Trends (Last 24 Hours)",
            xaxis_title="Time",
            yaxis_title="Concentration",
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, width='stretch')
    
    # Tab 6: Analytics
    with tab6:
        st.subheader("🤖 AI-Powered Correlation Analysis")
        
        X, y, feature_names = prepare_features(integrated_gdf, target_parameter=selected_parameter)
        
        if X is not None and len(X) > 5:
            model, metrics = train_regression_model(X, y)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Model R² Score", f"{metrics['r2_score']:.3f}")
            with col2:
                st.metric("Mean Absolute Error", f"{metrics['mae']:.3f}")
            
            insights = generate_insights(model, X, feature_names)
            
            st.subheader("Feature Importance")
            importance_df = pd.DataFrame({
                'Feature': insights['features'],
                'Coefficient': insights['coefficients'],
                'Absolute Impact': np.abs(insights['coefficients'])
            }).sort_values('Absolute Impact', ascending=False)
            
            fig = px.bar(
                importance_df,
                x='Absolute Impact',
                y='Feature',
                orientation='h',
                title="Feature Importance"
            )
            st.plotly_chart(fig, width='stretch')
        else:
            st.warning("Insufficient data for analysis")
    
    # Tab 7: Alerts & Notifications
    with tab7:
        st.subheader("🔔 Alert Subscriptions")
        
        if notifier:
            st.write("Subscribe to receive AQI alerts when pollution levels exceed your threshold")
            
            alert_type = st.radio("Notification Type:", ["Email", "SMS"])
            
            if alert_type == "Email":
                email = st.text_input("Email Address")
                threshold = st.slider("Alert Threshold (PM2.5 µg/m³)", 30, 250, 90)
                vulnerable = st.selectbox("Profile", ["general", "elderly", "children", "asthmatic"])
                
                if st.button("Subscribe to Email Alerts"):
                    if email:
                        # Get all locations from current data
                        all_locations = integrated_gdf['location_name'].unique().tolist()
                        vulnerable_list = [vulnerable] if vulnerable != 'general' else []
                        
                        notifier.subscribe_email(
                            email=email,
                            locations=all_locations,  # Subscribe to all locations
                            threshold_aqi=threshold,
                            vulnerable_groups=vulnerable_list
                        )
                        st.success(f"✅ Subscribed! You'll receive alerts when PM2.5 > {threshold}")
                    else:
                        st.warning("Please enter email address")
            
            else:
                phone = st.text_input("Phone Number (with country code, e.g., +91)")
                threshold = st.slider("Alert Threshold (PM2.5 µg/m³)", 30, 250, 90, key="sms_threshold")
                vulnerable = st.selectbox("Profile", ["general", "elderly", "children", "asthmatic"], key="sms_profile")
                
                if st.button("Subscribe to SMS Alerts"):
                    if phone:
                        # Get all locations from current data
                        all_locations = integrated_gdf['location_name'].unique().tolist()
                        
                        notifier.subscribe_sms(
                            phone=phone,
                            locations=all_locations,  # Subscribe to all locations
                            threshold_aqi=threshold
                        )
                        st.success(f"✅ Subscribed! You'll receive SMS when PM2.5 > {threshold}")
                    else:
                        st.warning("Please enter phone number")
            
            st.markdown("---")
            st.caption("💡 Alerts are sent when forecast predicts AQI above your threshold")
        else:
            st.warning("Notification service not available. Configure SendGrid/Twilio API keys.")
    
    # Tab 8: Data Table
    with tab8:
        st.subheader("Complete Dataset")
        
        display_df = integrated_gdf.drop(columns=['geometry'], errors='ignore')
        
        # Limit rows displayed to prevent memory issues
        max_rows = 1000
        if len(display_df) > max_rows:
            st.info(f"Showing first {max_rows} of {len(display_df)} rows")
            display_df = display_df.head(max_rows)
        
        st.dataframe(display_df, width='stretch', height=500)
        
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"delhi_env_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()