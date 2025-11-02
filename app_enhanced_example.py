"""
Enhanced Streamlit Dashboard - Integration Example
This file shows how to integrate all the advanced features into your main app.py

Copy sections from here into your app.py as needed.
"""

import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Import our custom modules
from config import config
from geocoding import GeocodingService
from ml_models.lstm_forecaster import AQIForecaster
from model_tracker import ModelTracker
from health_advisor import HealthAdvisor
from notifications import NotificationService
from dataloader import fetch_openaq_data, fetch_weather_data, load_geospatial_boundaries

# Initialize services
@st.cache_resource
def init_services():
    """Initialize all services once"""
    return {
        'geocoder': GeocodingService(),
        'forecaster': AQIForecaster(),
        'tracker': ModelTracker(),
        'advisor': HealthAdvisor(),
        'notifier': NotificationService()
    }

def render_location_search(services):
    """
    Render location search widget
    Returns: location details dict or None
    """
    st.subheader("🔍 Find Your Location")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input(
            "Enter address or pincode",
            placeholder="e.g., Connaught Place, Delhi or 110001",
            help="Search by landmark, street, area name, or 6-digit pincode"
        )
    
    with col2:
        use_geolocation = st.button("📍 Use My Location")
    
    location_details = None
    
    if search_query:
        with st.spinner("Searching..."):
            location_details = services['geocoder'].search_location(search_query)
            
            if location_details:
                st.success(f"✅ Found: {location_details.get('formatted_address', 'Location found')}")
                
                # Display location details
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Ward", location_details.get('ward_name', 'N/A'))
                with col2:
                    st.metric("District", location_details.get('district_name', 'N/A'))
                with col3:
                    st.metric("Coordinates", 
                             f"{location_details['latitude']:.4f}, {location_details['longitude']:.4f}")
            else:
                st.error("❌ Location not found. Please try a different query.")
    
    if use_geolocation:
        st.info("📱 Browser geolocation requires HTTPS. Using default: Connaught Place")
        # In production with HTTPS, you'd use JavaScript geolocation API
        location_details = services['geocoder'].search_location("Connaught Place, Delhi")
    
    return location_details

def render_forecast_tab(services, location_details=None):
    """Render the forecasting tab"""
    st.header("📊 Next 24-Hour AQI Forecast")
    
    # Load or use default location
    if location_details is None:
        st.info("Using default location: Anand Vihar. Search for your location above!")
        location_name = "Anand Vihar"
        lat, lon = 28.6469, 77.3160
    else:
        location_name = location_details.get('ward_name', 'Your Location')
        lat = location_details['latitude']
        lon = location_details['longitude']
    
    # Load model
    try:
        services['forecaster'].load_model(version='latest')
        
        # Get recent data (in production, fetch from database)
        # For demo, create sample data
        recent_data = create_sample_recent_data(location_name, lat, lon)
        
        # Make prediction
        forecast = services['forecaster'].predict(recent_data, steps_ahead=24)
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'hour': [(datetime.now() + timedelta(hours=i)).strftime('%I %p') 
                    for i in range(1, 25)],
            'aqi': forecast,
            'category': [config.get_aqi_category(aqi)['category'] for aqi in forecast],
            'color': [config.get_aqi_category(aqi)['color'] for aqi in forecast]
        })
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current AQI", f"{forecast[0]:.0f}")
        with col2:
            peak_aqi = max(forecast)
            st.metric("Peak (24h)", f"{peak_aqi:.0f}", 
                     delta=f"+{peak_aqi - forecast[0]:.0f}")
        with col3:
            avg_aqi = sum(forecast) / len(forecast)
            st.metric("Average (24h)", f"{avg_aqi:.0f}")
        with col4:
            peak_hour = forecast.index(peak_aqi) + 1
            peak_time = (datetime.now() + timedelta(hours=peak_hour)).strftime('%I %p')
            st.metric("Peak Time", peak_time)
        
        # Plot forecast
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=forecast_df['hour'],
            y=forecast_df['aqi'],
            mode='lines+markers',
            name='Forecasted AQI',
            line=dict(color='#ff6b6b', width=3),
            marker=dict(size=8, color=forecast_df['color']),
            hovertemplate='<b>%{x}</b><br>AQI: %{y:.0f}<extra></extra>'
        ))
        
        # Add category bands
        fig.add_hrect(y0=0, y1=50, fillcolor="green", opacity=0.1, line_width=0)
        fig.add_hrect(y0=51, y1=100, fillcolor="yellow", opacity=0.1, line_width=0)
        fig.add_hrect(y0=101, y1=200, fillcolor="orange", opacity=0.1, line_width=0)
        fig.add_hrect(y0=201, y1=300, fillcolor="red", opacity=0.1, line_width=0)
        fig.add_hrect(y0=301, y1=500, fillcolor="purple", opacity=0.1, line_width=0)
        
        fig.update_layout(
            title=f"24-Hour AQI Forecast for {location_name}",
            xaxis_title="Time",
            yaxis_title="AQI",
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Health advisory
        st.subheader("💊 Health Advisory")
        
        vulnerable_groups = st.multiselect(
            "Select if you belong to any vulnerable group:",
            options=['elderly', 'children', 'asthmatic', 'heart_disease', 'pregnant'],
            format_func=lambda x: {
                'elderly': '👵 Elderly (65+)',
                'children': '👶 Children',
                'asthmatic': '🫁 Asthma Patient',
                'heart_disease': '❤️ Heart Disease',
                'pregnant': '🤰 Pregnant'
            }[x]
        )
        
        advisory = services['advisor'].get_advisory(
            forecast[0],
            vulnerable_groups=vulnerable_groups if vulnerable_groups else None
        )
        
        # Display advisory
        st.markdown(f"""
        <div style="padding: 20px; background-color: {advisory['color']}20; border-left: 5px solid {advisory['color']}; border-radius: 5px;">
            <h3>{advisory['icon']} {advisory['category']}</h3>
            <p><strong>{advisory['summary']}</strong></p>
            <p>{advisory['health_impact']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📋 Recommendations")
        for rec in advisory['recommendations']:
            st.markdown(f"- {rec}")
        
        # Vulnerable group specific advice
        if vulnerable_groups and 'vulnerable_groups' in advisory:
            st.markdown("### ⚠️ Special Precautions for You")
            for group, info in advisory['vulnerable_groups'].items():
                with st.expander(f"{info['icon']} {info['name']}"):
                    st.markdown(f"**Risk Level:** {info['risk_level']}")
                    st.markdown("**Risk Factors:**")
                    for factor in info['risk_factors']:
                        st.markdown(f"- {factor}")
                    st.markdown("**Specific Advice:**")
                    for advice in info['specific_advice']:
                        st.markdown(f"- {advice}")
        
        # WhatsApp share button
        st.markdown("### 📱 Share This Forecast")
        whatsapp_msg = services['advisor'].get_whatsapp_message(
            forecast[0],
            location_name,
            "https://delhiaqi.app"  # Replace with your actual URL
        )
        whatsapp_url = f"https://wa.me/?text={whatsapp_msg.replace(' ', '%20').replace('\n', '%0A')}"
        
        st.markdown(f'<a href="{whatsapp_url}" target="_blank" style="display: inline-block; padding: 10px 20px; background-color: #25D366; color: white; text-decoration: none; border-radius: 5px;">📱 Share on WhatsApp</a>', unsafe_allow_html=True)
        
    except FileNotFoundError:
        st.warning("⚠️ Model not trained yet. Please run the training pipeline first.")
        st.code("python ml_models/lstm_forecaster.py")

def render_model_performance_tab(services):
    """Render model performance monitoring tab"""
    st.header("🤖 Model Performance & Accuracy")
    
    # Get accuracy badge
    badge = services['tracker'].generate_accuracy_badge()
    
    # Display badge
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        badge_color = {
            'green': '#00e400',
            'yellow': '#ffff00',
            'orange': '#ff7e00',
            'red': '#ff0000',
            'gray': '#888888'
        }[badge['color']]
        
        st.markdown(f"""
        <div style="text-align: center; padding: 30px; background-color: {badge_color}20; border: 3px solid {badge_color}; border-radius: 10px;">
            <h1 style="color: {badge_color}; font-size: 48px; margin: 0;">{badge['accuracy']}</h1>
            <p style="font-size: 18px; margin: 10px 0;">{badge['label']}</p>
            <p style="font-size: 14px; color: #666;">Status: {badge['status'].title()}</p>
            <p style="font-size: 12px; color: #888;">Based on {badge.get('n_comparisons', 0)} comparisons</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Get model progress
    progress = services['tracker'].get_model_progress()
    
    if progress.get('status') != 'no_data':
        st.subheader("📈 Model Improvement Progress")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("First Prediction", progress['first_prediction_date'])
        with col2:
            st.metric("Last Prediction", progress['last_prediction_date'])
        with col3:
            st.metric("Total Predictions", progress['total_predictions'])
        
        # Accuracy comparison
        st.subheader("🎯 Accuracy Trends")
        
        acc_7d = progress.get('accuracy_7_days', {})
        acc_30d = progress.get('accuracy_30_days', {})
        
        if acc_7d.get('status') == 'success' and acc_30d.get('status') == 'success':
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "Last 7 Days",
                    f"{acc_7d['accuracy_percentage']:.1f}%",
                    delta=f"{acc_7d['accuracy_percentage'] - acc_30d['accuracy_percentage']:.1f}%"
                )
                st.caption(f"MAE: {acc_7d['mae']:.2f} | RMSE: {acc_7d['rmse']:.2f}")
            
            with col2:
                st.metric(
                    "Last 30 Days",
                    f"{acc_30d['accuracy_percentage']:.1f}%"
                )
                st.caption(f"MAE: {acc_30d['mae']:.2f} | RMSE: {acc_30d['rmse']:.2f}")
        
        # Prediction vs Actual Chart
        st.subheader("📊 Predictions vs Actual Values")
        
        comparison_df = services['tracker'].get_prediction_vs_actual_data(days_back=7)
        
        if not comparison_df.empty:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=comparison_df['timestamp'],
                y=comparison_df['predicted'],
                mode='lines',
                name='Predicted',
                line=dict(color='#3498db', width=2, dash='dash')
            ))
            
            fig.add_trace(go.Scatter(
                x=comparison_df['timestamp'],
                y=comparison_df['actual'],
                mode='markers',
                name='Actual',
                marker=dict(color='#e74c3c', size=6)
            ))
            
            fig.update_layout(
                title="Model Predictions vs Actual Observations (Last 7 Days)",
                xaxis_title="Timestamp",
                yaxis_title="AQI",
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data yet to show comparison chart. Model needs to make predictions and collect actual values.")
    
    else:
        st.info("🔄 Model is still in training phase. Accuracy metrics will appear once predictions are compared with actual values.")

def render_alert_subscription_tab(services):
    """Render alert subscription form"""
    st.header("🔔 Subscribe to Air Quality Alerts")
    
    st.markdown("""
    Get personalized alerts when air quality is forecasted to deteriorate in your area.
    We'll notify you 6-12 hours in advance so you can plan accordingly.
    """)
    
    with st.form("subscription_form"):
        st.subheader("Email Alerts")
        
        email = st.text_input("Email Address", placeholder="your.email@example.com")
        
        locations = st.multiselect(
            "Monitor Locations",
            options=["Anand Vihar", "Punjabi Bagh", "RK Puram", "ITO", "Dwarka", "Rohini"],
            default=["Anand Vihar"],
            help="Select areas you want to monitor"
        )
        
        threshold = st.select_slider(
            "Alert Threshold",
            options=[50, 100, 150, 200, 250, 300],
            value=200,
            help="Receive alerts when AQI is forecasted to exceed this value",
            format_func=lambda x: f"{x} ({config.get_aqi_category(x)['category']})"
        )
        
        vulnerable_groups = st.multiselect(
            "I belong to these groups (optional):",
            options=['elderly', 'children', 'asthmatic', 'heart_disease', 'pregnant'],
            format_func=lambda x: {
                'elderly': '👵 Elderly (65+)',
                'children': '👶 Children',
                'asthmatic': '🫁 Asthma Patient',
                'heart_disease': '❤️ Heart Disease',
                'pregnant': '🤰 Pregnant'
            }[x],
            help="Get personalized health advisories"
        )
        
        submit = st.form_submit_button("Subscribe to Alerts")
        
        if submit:
            if email and locations:
                services['notifier'].subscribe_email(
                    email=email,
                    locations=locations,
                    threshold_aqi=threshold,
                    vulnerable_groups=vulnerable_groups
                )
                st.success(f"✅ Successfully subscribed {email} to alerts for {', '.join(locations)}")
            else:
                st.error("Please provide email and select at least one location")
    
    # Show subscriber count
    stats = services['notifier'].get_subscriber_count()
    st.info(f"📊 Currently {stats['total_active']} active subscribers")

def create_sample_recent_data(location_name, lat, lon):
    """Create sample recent data for demo purposes"""
    import numpy as np
    
    dates = pd.date_range(end=datetime.now(), periods=200, freq='H')
    
    data = {
        'timestamp': dates,
        'location_name': [location_name] * len(dates),
        'parameter': ['pm25'] * len(dates),
        'value': 100 + np.random.normal(0, 20, len(dates)),
        'latitude': [lat] * len(dates),
        'longitude': [lon] * len(dates),
        'temperature_c': 25 + np.random.normal(0, 5, len(dates)),
        'humidity_percent': 60 + np.random.normal(0, 10, len(dates)),
        'wind_speed_mps': 3 + np.random.normal(0, 1, len(dates)),
        'pressure_hpa': 1013 + np.random.normal(0, 5, len(dates)),
        'cloud_cover': np.random.uniform(0, 100, len(dates)),
        'precipitation_mm': np.zeros(len(dates)),
        'hour': dates.hour,
        'day_of_week': dates.dayofweek,
        'month': dates.month,
        'is_weekend': dates.dayofweek.isin([5, 6]).astype(int),
        'is_holiday': np.zeros(len(dates), dtype=int),
        'is_diwali': np.zeros(len(dates), dtype=int),
        'is_crop_burning_season': (dates.month.isin([10, 11, 12])).astype(int)
    }
    
    return pd.DataFrame(data)


# Example of how to integrate into main app
if __name__ == "__main__":
    st.set_page_config(page_title="Delhi AQI Dashboard", page_icon="🌍", layout="wide")
    
    services = init_services()
    
    st.title("🌍 Delhi Air Quality Dashboard - Advanced Features")
    
    # Location search (always visible)
    location_details = render_location_search(services)
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Forecast",
        "🤖 Model Performance",
        "🔔 Alerts",
        "📥 Export Data"
    ])
    
    with tab1:
        render_forecast_tab(services, location_details)
    
    with tab2:
        render_model_performance_tab(services)
    
    with tab3:
        render_alert_subscription_tab(services)
    
    with tab4:
        st.header("📥 Download Data")
        st.info("Data export functionality - see utils/export.py for implementation")
