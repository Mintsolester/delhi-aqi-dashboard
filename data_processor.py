"""
Data Processor Module
Handles data cleaning, transformation, and spatial joining operations
"""

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

def harmonize_aq_weather(aq_df, weather_dict):
    """
    Combine air quality data with weather data
    """
    integrated_df = aq_df.copy()
    
    # Add weather parameters
    integrated_df['temperature_c'] = weather_dict.get('temperature_c', None)
    integrated_df['humidity_percent'] = weather_dict.get('humidity_percent', None)
    integrated_df['wind_speed_mps'] = weather_dict.get('wind_speed_mps', None)
    integrated_df['pressure_hpa'] = weather_dict.get('pressure_hpa', None)
    
    # Convert timestamp to datetime
    if integrated_df['timestamp'].dtype == 'object':
        integrated_df['timestamp'] = pd.to_datetime(integrated_df['timestamp'])
    
    # Add temporal features
    integrated_df['hour'] = integrated_df['timestamp'].dt.hour
    integrated_df['day_of_week'] = integrated_df['timestamp'].dt.dayofweek
    
    return integrated_df

def perform_spatial_join(points_df, polygons_gdf, target_column='ward_name'):
    """
    Perform spatial join between point data and polygon boundaries
    """
    # Convert to GeoDataFrame
    geometry = [Point(xy) for xy in zip(points_df['longitude'], points_df['latitude'])]
    points_gdf = gpd.GeoDataFrame(points_df, geometry=geometry, crs='EPSG:4326')
    
    # Ensure same CRS
    if polygons_gdf.crs != points_gdf.crs:
        polygons_gdf = polygons_gdf.to_crs(points_gdf.crs)
    
    # Spatial join
    joined_gdf = gpd.sjoin(points_gdf, polygons_gdf, how='left', predicate='within')
    
    # Rename columns
    if 'name' in joined_gdf.columns:
        joined_gdf = joined_gdf.rename(columns={'name': target_column})
    
    # Clean up duplicate columns
    cols_to_keep = [col for col in joined_gdf.columns if not col.endswith('_right')]
    joined_gdf = joined_gdf[cols_to_keep]
    
    return joined_gdf

def aggregate_by_geography(integrated_gdf, geography_level='ward', agg_func='mean'):
    """
    Aggregate pollutant data by geographic area
    """
    if geography_level.lower() == 'ward':
        group_column = 'ward_name'
    elif geography_level.lower() == 'district':
        group_column = 'district_name'
    else:
        group_column = 'location_name'
    
    if group_column not in integrated_gdf.columns:
        group_column = 'location_name'
    
    agg_dict = {
        'value': 'mean',
        'latitude': 'first',
        'longitude': 'first'
    }
    
    agg_df = integrated_gdf.groupby([group_column, 'parameter']).agg(agg_dict).reset_index()
    
    if isinstance(agg_df.columns, pd.MultiIndex):
        agg_df.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                         for col in agg_df.columns.values]
    
    return agg_df
