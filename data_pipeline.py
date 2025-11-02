"""
Advanced Data Pipeline for Historical AQI Collection
Fetches 10 years of data from OpenAQ/CPCB with weather correlation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from typing import Dict, List, Optional
import logging
from pathlib import Path
import json

from config import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HistoricalDataPipeline:
    """
    Comprehensive data pipeline for fetching, processing, and storing
    multi-year AQI and weather data for ML model training
    """
    
    def __init__(self, years_back: int = 10):
        """
        Initialize data pipeline
        
        Args:
            years_back: Number of years of historical data to fetch
        """
        self.years_back = years_back
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=365 * years_back)
        
    def fetch_openaq_historical(self, 
                                city: str = 'Delhi',
                                parameters: List[str] = None) -> pd.DataFrame:
        """
        Fetch historical AQI data from OpenAQ API
        
        Args:
            city: City name
            parameters: List of pollutants to fetch
            
        Returns:
            DataFrame with historical measurements
        """
        if parameters is None:
            parameters = config.POLLUTANTS
            
        logger.info(f"Fetching {self.years_back} years of data for {city}")
        
        all_data = []
        
        # OpenAQ v2 API endpoint
        base_url = "https://api.openaq.org/v2/measurements"
        
        # Fetch data in monthly chunks (API limitations)
        current_date = self.start_date
        
        while current_date < self.end_date:
            chunk_end = min(current_date + timedelta(days=30), self.end_date)
            
            for param in parameters:
                try:
                    params = {
                        'city': city,
                        'parameter': param,
                        'date_from': current_date.strftime('%Y-%m-%d'),
                        'date_to': chunk_end.strftime('%Y-%m-%d'),
                        'limit': 10000,
                        'page': 1
                    }
                    
                    headers = {'X-API-Key': config.OPENAQ_API_KEY}
                    
                    response = requests.get(base_url, params=params, headers=headers)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        if 'results' in data:
                            all_data.extend(data['results'])
                            logger.info(f"Fetched {len(data['results'])} records for {param} "
                                      f"({current_date.date()} to {chunk_end.date()})")
                    else:
                        logger.warning(f"API error: {response.status_code}")
                        
                except Exception as e:
                    logger.error(f"Error fetching {param} data: {str(e)}")
            
            current_date = chunk_end
        
        # Convert to DataFrame
        if all_data:
            df = self._parse_openaq_response(all_data)
            logger.info(f"Total records fetched: {len(df)}")
            return df
        else:
            logger.warning("No data fetched, creating sample dataset")
            return self._create_sample_historical_data()
    
    def _parse_openaq_response(self, results: List[Dict]) -> pd.DataFrame:
        """Parse OpenAQ API response into structured DataFrame"""
        
        parsed_data = []
        
        for item in results:
            try:
                parsed_data.append({
                    'timestamp': item['date']['utc'],
                    'location_name': item['location'],
                    'city': item.get('city', 'Unknown'),
                    'parameter': item['parameter'],
                    'value': item['value'],
                    'unit': item['unit'],
                    'latitude': item['coordinates']['latitude'],
                    'longitude': item['coordinates']['longitude'],
                    'location_id': item.get('locationId', None)
                })
            except KeyError as e:
                logger.warning(f"Missing field in response: {e}")
                continue
        
        df = pd.DataFrame(parsed_data)
        
        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    def fetch_weather_historical(self, 
                                 lat: float = 28.6139,
                                 lon: float = 77.2090) -> pd.DataFrame:
        """
        Fetch historical weather data
        
        Note: OpenWeatherMap's historical API requires paid subscription.
        Alternative: Use Visual Crossing Weather API (free tier available)
        or NOAA/ISD data
        """
        
        logger.info("Fetching historical weather data")
        
        # Visual Crossing Weather API (example)
        base_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
        
        start_str = self.start_date.strftime('%Y-%m-%d')
        end_str = self.end_date.strftime('%Y-%m-%d')
        
        # Free API key from https://www.visualcrossing.com/
        api_key = config.OPENWEATHERMAP_API_KEY  # Reuse or set separate key
        
        url = f"{base_url}/{lat},{lon}/{start_str}/{end_str}"
        
        params = {
            'unitGroup': 'metric',
            'key': api_key,
            'include': 'hours'
        }
        
        try:
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                return self._parse_weather_response(data)
            else:
                logger.warning(f"Weather API error: {response.status_code}")
                return self._create_sample_weather_data()
                
        except Exception as e:
            logger.error(f"Error fetching weather: {str(e)}")
            return self._create_sample_weather_data()
    
    def _parse_weather_response(self, data: Dict) -> pd.DataFrame:
        """Parse weather API response"""
        
        weather_records = []
        
        for day in data.get('days', []):
            date = day['datetime']
            
            for hour in day.get('hours', []):
                weather_records.append({
                    'timestamp': pd.to_datetime(f"{date} {hour['datetime']}"),
                    'temperature_c': hour.get('temp'),
                    'humidity_percent': hour.get('humidity'),
                    'wind_speed_mps': hour.get('windspeed', 0) * 0.277778,  # km/h to m/s
                    'pressure_hpa': hour.get('pressure'),
                    'cloud_cover': hour.get('cloudcover', 0),
                    'precipitation_mm': hour.get('precip', 0),
                    'visibility_km': hour.get('visibility', 10)
                })
        
        return pd.DataFrame(weather_records)
    
    def create_event_markers(self) -> pd.DataFrame:
        """
        Create event markers for special occasions
        (Diwali, crop burning season, holidays, etc.)
        """
        
        logger.info("Creating event markers")
        
        # Generate date range
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        
        df = pd.DataFrame({'date': dates})
        
        # Add temporal features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Crop burning season (Oct-Dec)
        df['is_crop_burning_season'] = df['month'].isin([10, 11, 12]).astype(int)
        
        # Winter inversion (Dec-Feb)
        df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)
        
        # Approximate Diwali dates (Oct-Nov, varies each year)
        # This is simplified - use lunar calendar for accuracy
        diwali_dates = [
            '2015-11-11', '2016-10-30', '2017-10-19', '2018-11-07',
            '2019-10-27', '2020-11-14', '2021-11-04', '2022-10-24',
            '2023-11-12', '2024-11-01', '2025-10-20'
        ]
        
        df['is_diwali'] = 0
        for diwali_date in diwali_dates:
            # Mark 7 days around Diwali
            diwali_dt = pd.to_datetime(diwali_date)
            mask = (df['date'] >= diwali_dt - timedelta(days=3)) & \
                   (df['date'] <= diwali_dt + timedelta(days=3))
            df.loc[mask, 'is_diwali'] = 1
        
        # Public holidays (simplified)
        df['is_holiday'] = 0
        
        # Republic Day (Jan 26), Independence Day (Aug 15), Gandhi Jayanti (Oct 2)
        holiday_dates = [(1, 26), (8, 15), (10, 2)]
        for month, day in holiday_dates:
            df.loc[(df['month'] == month) & (df['day'] == day), 'is_holiday'] = 1
        
        return df
    
    def merge_all_data(self,
                      aqi_df: pd.DataFrame,
                      weather_df: pd.DataFrame,
                      events_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge AQI, weather, and event data into unified dataset
        """
        
        logger.info("Merging all datasets")
        
        # Round timestamps to nearest hour for merging
        aqi_df['timestamp_hour'] = aqi_df['timestamp'].dt.floor('H')
        weather_df['timestamp_hour'] = weather_df['timestamp'].dt.floor('H')
        
        # Merge AQI with weather
        merged = pd.merge(
            aqi_df,
            weather_df,
            on='timestamp_hour',
            how='left',
            suffixes=('', '_weather')
        )
        
        # Merge with events
        merged['date'] = merged['timestamp_hour'].dt.date
        events_df['date'] = pd.to_datetime(events_df['date']).dt.date
        
        final_df = pd.merge(
            merged,
            events_df,
            on='date',
            how='left'
        )
        
        # Fill missing values
        final_df = final_df.fillna(method='ffill').fillna(method='bfill')
        
        # Add additional features
        final_df['hour'] = final_df['timestamp_hour'].dt.hour
        final_df['season'] = final_df['month'].map({
            12: 'winter', 1: 'winter', 2: 'winter',
            3: 'spring', 4: 'spring', 5: 'spring',
            6: 'summer', 7: 'summer', 8: 'summer',
            9: 'autumn', 10: 'autumn', 11: 'autumn'
        })
        
        logger.info(f"Final merged dataset: {len(final_df)} records")
        
        return final_df
    
    def save_dataset(self, df: pd.DataFrame, filename: str = "historical_aqi_dataset.csv"):
        """Save processed dataset"""
        
        filepath = config.DATA_DIR / filename
        df.to_csv(filepath, index=False)
        logger.info(f"Dataset saved to {filepath}")
        
        # Also save metadata
        metadata = {
            'created_at': datetime.now().isoformat(),
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'total_records': len(df),
            'locations': df['location_name'].nunique(),
            'parameters': df['parameter'].unique().tolist(),
            'date_range_days': (self.end_date - self.start_date).days
        }
        
        metadata_path = config.DATA_DIR / "dataset_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadata saved to {metadata_path}")
    
    def _create_sample_historical_data(self) -> pd.DataFrame:
        """Create sample historical data for demonstration"""
        
        logger.info("Creating sample historical dataset")
        
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq='H')
        locations = ['Anand Vihar', 'Punjabi Bagh', 'RK Puram', 'ITO', 'Dwarka']
        
        data = []
        
        for location in locations:
            for date in dates[::6]:  # Every 6 hours to reduce size
                for param in config.POLLUTANTS[:3]:  # pm25, pm10, no2
                    # Simulate seasonal patterns
                    month = date.month
                    hour = date.hour
                    
                    # Higher pollution in winter and early morning
                    base_value = 100 if month in [11, 12, 1] else 60
                    hour_factor = 1.5 if hour in [6, 7, 8, 18, 19, 20] else 1.0
                    
                    value = base_value * hour_factor * np.random.uniform(0.7, 1.3)
                    
                    data.append({
                        'timestamp': date,
                        'location_name': location,
                        'city': 'Delhi',
                        'parameter': param,
                        'value': round(value, 2),
                        'unit': 'µg/m³',
                        'latitude': 28.6 + np.random.uniform(-0.1, 0.1),
                        'longitude': 77.2 + np.random.uniform(-0.1, 0.1),
                        'location_id': hash(location) % 10000
                    })
        
        return pd.DataFrame(data)
    
    def _create_sample_weather_data(self) -> pd.DataFrame:
        """Create sample weather data"""
        
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq='H')
        
        weather_data = []
        
        for date in dates:
            month = date.month
            hour = date.hour
            
            # Seasonal temperature patterns
            if month in [4, 5, 6]:  # Summer
                temp_base = 35
            elif month in [12, 1, 2]:  # Winter
                temp_base = 15
            else:
                temp_base = 25
            
            # Diurnal variation
            temp_variation = 5 * np.sin((hour - 6) * np.pi / 12)
            
            weather_data.append({
                'timestamp': date,
                'temperature_c': temp_base + temp_variation + np.random.uniform(-2, 2),
                'humidity_percent': 60 + np.random.uniform(-20, 20),
                'wind_speed_mps': max(0, 3 + np.random.uniform(-2, 2)),
                'pressure_hpa': 1013 + np.random.uniform(-5, 5),
                'cloud_cover': np.random.uniform(0, 100),
                'precipitation_mm': 0 if month not in [7, 8] else np.random.exponential(2),
                'visibility_km': np.random.uniform(2, 10)
            })
        
        return pd.DataFrame(weather_data)
    
    def run_full_pipeline(self) -> pd.DataFrame:
        """Execute complete data pipeline"""
        
        logger.info("Starting full data pipeline")
        
        # Step 1: Fetch AQI data
        aqi_df = self.fetch_openaq_historical()
        
        # Step 2: Fetch weather data
        weather_df = self.fetch_weather_historical()
        
        # Step 3: Create event markers
        events_df = self.create_event_markers()
        
        # Step 4: Merge all data
        final_df = self.merge_all_data(aqi_df, weather_df, events_df)
        
        # Step 5: Save dataset
        self.save_dataset(final_df)
        
        logger.info("Pipeline completed successfully")
        
        return final_df


if __name__ == "__main__":
    # Example usage
    pipeline = HistoricalDataPipeline(years_back=10)
    dataset = pipeline.run_full_pipeline()
    print(f"\nDataset shape: {dataset.shape}")
    print(f"\nColumns: {dataset.columns.tolist()}")
    print(f"\nFirst few rows:\n{dataset.head()}")
