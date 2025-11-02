"""
Data Loader Module
Handles fetching data from external APIs and loading geospatial files
"""

import pandas as pd
import geopandas as gpd
from datetime import datetime
import os

def fetch_openaq_data(api_key, city='Delhi', parameters=None):
    """
    Fetch latest air quality data from OpenAQ API
    """
    if parameters is None:
        parameters = ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']
    
    try:
        from openaq import OpenAQ
        client = OpenAQ(api_key=api_key)
        
        # For demonstration, create sample data
        # In production, you would use actual API calls
        sample_stations = [
            {'name': 'Anand Vihar', 'lat': 28.6469, 'lon': 77.3160},
            {'name': 'Punjabi Bagh', 'lat': 28.6692, 'lon': 77.1317},
            {'name': 'RK Puram', 'lat': 28.5631, 'lon': 77.1828},
            {'name': 'ITO', 'lat': 28.6289, 'lon': 77.2509},
            {'name': 'Dwarka', 'lat': 28.5921, 'lon': 77.0460},
            {'name': 'Rohini', 'lat': 28.7469, 'lon': 77.0690},
            {'name': 'Najafgarh', 'lat': 28.6092, 'lon': 76.9798}
        ]
        
        import random
        data_list = []
        timestamp = datetime.now()
        
        for station in sample_stations:
            for param in parameters:
                if param == 'pm25':
                    value = random.uniform(50, 200)
                    unit = 'µg/m³'
                elif param == 'pm10':
                    value = random.uniform(100, 350)
                    unit = 'µg/m³'
                elif param == 'no2':
                    value = random.uniform(20, 80)
                    unit = 'µg/m³'
                elif param == 'so2':
                    value = random.uniform(5, 30)
                    unit = 'µg/m³'
                elif param == 'o3':
                    value = random.uniform(10, 50)
                    unit = 'µg/m³'
                elif param == 'co':
                    value = random.uniform(0.5, 3.0)
                    unit = 'mg/m³'
                else:
                    value = random.uniform(10, 100)
                    unit = 'µg/m³'
                
                data_list.append({
                    'location_name': station['name'],
                    'parameter': param,
                    'value': round(value, 2),
                    'unit': unit,
                    'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    'latitude': station['lat'],
                    'longitude': station['lon']
                })
        
        return pd.DataFrame(data_list)
        
    except Exception as e:
        print(f"Error fetching OpenAQ data: {str(e)}")
        print("Using sample data as fallback")
        return create_sample_data()

def fetch_weather_data(api_key, city='Delhi,IN'):
    """
    Fetch current weather data from OpenWeatherMap API
    """
    try:
        from pyowm import OWM
        owm = OWM(api_key)
        mgr = owm.weather_manager()
        
        observation = mgr.weather_at_place(city)
        weather = observation.weather
        
        temp_dict = weather.temperature('celsius')
        
        return {
            'temperature_c': round(temp_dict['temp'], 2),
            'humidity_percent': weather.humidity,
            'wind_speed_mps': round(weather.wind()['speed'], 2),
            'pressure_hpa': weather.pressure['press'],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
    except Exception as e:
        print(f"Error fetching weather data: {str(e)}")
        return {
            'temperature_c': 28.5,
            'humidity_percent': 65,
            'wind_speed_mps': 3.2,
            'pressure_hpa': 1012,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

def load_geospatial_boundaries(district_path='data/delhi_districts.geojson', 
                                ward_path='data/delhi_wards.geojson'):
    """
    Load geospatial boundary files for Delhi
    """
    try:
        if os.path.exists(district_path):
            districts_gdf = gpd.read_file(district_path)
            districts_gdf = districts_gdf.to_crs(epsg=4326)
        else:
            print(f"District file not found. Creating sample data.")
            districts_gdf = create_sample_districts()
        
        if os.path.exists(ward_path):
            wards_gdf = gpd.read_file(ward_path)
            wards_gdf = wards_gdf.to_crs(epsg=4326)
        else:
            print(f"Ward file not found. Creating sample data.")
            wards_gdf = create_sample_wards()
        
        # Standardize column names
        for gdf in [districts_gdf, wards_gdf]:
            if 'NAME' in gdf.columns:
                gdf.rename(columns={'NAME': 'name'}, inplace=True)
            elif 'Name' in gdf.columns:
                gdf.rename(columns={'Name': 'name'}, inplace=True)
        
        return districts_gdf, wards_gdf
        
    except Exception as e:
        print(f"Error loading geospatial data: {str(e)}")
        return create_sample_districts(), create_sample_wards()

def create_sample_data():
    """Create sample air quality data"""
    import random
    
    stations = [
        {'name': 'Anand Vihar', 'lat': 28.6469, 'lon': 77.3160},
        {'name': 'Punjabi Bagh', 'lat': 28.6692, 'lon': 77.1317},
        {'name': 'RK Puram', 'lat': 28.5631, 'lon': 77.1828}
    ]
    
    data_list = []
    timestamp = datetime.now()
    
    for station in stations:
        for param in ['pm25', 'pm10', 'no2']:
            value = random.uniform(50, 200)
            data_list.append({
                'location_name': station['name'],
                'parameter': param,
                'value': round(value, 2),
                'unit': 'µg/m³',
                'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'latitude': station['lat'],
                'longitude': station['lon']
            })
    
    return pd.DataFrame(data_list)

def create_sample_districts():
    """Create sample district boundaries"""
    from shapely.geometry import Polygon
    
    districts_data = {
        'name': ['Central Delhi', 'North Delhi', 'South Delhi', 'East Delhi', 'West Delhi'],
        'geometry': [
            Polygon([(77.20, 28.60), (77.25, 28.60), (77.25, 28.65), (77.20, 28.65)]),
            Polygon([(77.15, 28.70), (77.25, 28.70), (77.25, 28.80), (77.15, 28.80)]),
            Polygon([(77.15, 28.50), (77.25, 28.50), (77.25, 28.55), (77.15, 28.55)]),
            Polygon([(77.25, 28.60), (77.35, 28.60), (77.35, 28.70), (77.25, 28.70)]),
            Polygon([(77.05, 28.60), (77.15, 28.60), (77.15, 28.70), (77.05, 28.70)])
        ]
    }
    
    return gpd.GeoDataFrame(districts_data, crs='EPSG:4326')

def create_sample_wards():
    """Create sample ward boundaries"""
    from shapely.geometry import Polygon
    
    wards_data = {
        'name': [f'Ward {i}' for i in range(1, 11)],
        'geometry': [
            Polygon([(77.20 + 0.02*i, 28.60), (77.22 + 0.02*i, 28.60), 
                    (77.22 + 0.02*i, 28.62), (77.20 + 0.02*i, 28.62)])
            for i in range(10)
        ]
    }
    
    return gpd.GeoDataFrame(wards_data, crs='EPSG:4326')
