"""
Geocoding Service
Location search, address resolution, and ward-level mapping
"""

import pandas as pd
import geopandas as gpd
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from shapely.geometry import Point
import logging
from typing import Dict, Optional, Tuple, List
import re

from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeocodingService:
    """
    Service for geocoding addresses, pincodes, and reverse geocoding
    Maps coordinates to administrative boundaries (ward/district)
    """
    
    def __init__(self):
        """Initialize geocoding service"""
        self.geolocator = Nominatim(user_agent="delhi_aqi_dashboard")
        self.wards_gdf = None
        self.districts_gdf = None
        self._load_boundaries()
    
    def _load_boundaries(self):
        """Load ward and district boundaries"""
        try:
            # Load district boundaries
            district_path = config.DATA_DIR / "delhi_districts.geojson"
            if district_path.exists():
                self.districts_gdf = gpd.read_file(district_path)
                self.districts_gdf = self.districts_gdf.to_crs(epsg=4326)
                logger.info(f"Loaded {len(self.districts_gdf)} districts")
            
            # Load ward boundaries (assembly constituencies)
            ward_path = config.DATA_DIR / "delhi_assembly-constituencie.geojson"
            if ward_path.exists():
                self.wards_gdf = gpd.read_file(ward_path)
                self.wards_gdf = self.wards_gdf.to_crs(epsg=4326)
                logger.info(f"Loaded {len(self.wards_gdf)} wards")
            
            # Standardize column names
            for gdf in [self.districts_gdf, self.wards_gdf]:
                if gdf is not None:
                    if 'NAME' in gdf.columns:
                        gdf.rename(columns={'NAME': 'name'}, inplace=True)
                    elif 'Name' in gdf.columns:
                        gdf.rename(columns={'Name': 'name'}, inplace=True)
                    elif 'AC_NAME' in gdf.columns:
                        gdf.rename(columns={'AC_NAME': 'name'}, inplace=True)
        
        except Exception as e:
            logger.error(f"Error loading boundaries: {str(e)}")
    
    def geocode_address(self, address: str) -> Optional[Dict]:
        """
        Geocode an address to coordinates
        
        Args:
            address: Full address or partial address
            
        Returns:
            Dict with lat, lon, and formatted address
        """
        try:
            # Add Delhi to query if not present
            if 'delhi' not in address.lower():
                address = f"{address}, Delhi, India"
            
            location = self.geolocator.geocode(address, timeout=10)
            
            if location:
                result = {
                    'latitude': location.latitude,
                    'longitude': location.longitude,
                    'formatted_address': location.address,
                    'raw': location.raw
                }
                
                logger.info(f"Geocoded '{address}' to {location.latitude}, {location.longitude}")
                return result
            else:
                logger.warning(f"Could not geocode address: {address}")
                return None
                
        except (GeocoderTimedOut, GeocoderServiceError) as e:
            logger.error(f"Geocoding error: {str(e)}")
            return None
    
    def geocode_pincode(self, pincode: str) -> Optional[Dict]:
        """
        Geocode Delhi pincode to coordinates
        
        Args:
            pincode: 6-digit pincode
            
        Returns:
            Dict with lat, lon, and area name
        """
        # Validate pincode format
        if not re.match(r'^\d{6}$', pincode):
            logger.warning(f"Invalid pincode format: {pincode}")
            return None
        
        # Delhi pincodes start with 110
        if not pincode.startswith('110'):
            logger.warning(f"Pincode {pincode} is not in Delhi")
            return None
        
        query = f"{pincode}, Delhi, India"
        return self.geocode_address(query)
    
    def reverse_geocode(self, latitude: float, longitude: float) -> Optional[Dict]:
        """
        Reverse geocode coordinates to address
        
        Args:
            latitude: Latitude
            longitude: Longitude
            
        Returns:
            Dict with address components
        """
        try:
            location = self.geolocator.reverse(
                (latitude, longitude),
                timeout=10,
                language='en'
            )
            
            if location:
                address = location.raw.get('address', {})
                
                result = {
                    'formatted_address': location.address,
                    'suburb': address.get('suburb', ''),
                    'neighbourhood': address.get('neighbourhood', ''),
                    'city': address.get('city', address.get('state_district', '')),
                    'state': address.get('state', ''),
                    'postcode': address.get('postcode', ''),
                    'latitude': latitude,
                    'longitude': longitude
                }
                
                return result
            else:
                return None
                
        except Exception as e:
            logger.error(f"Reverse geocoding error: {str(e)}")
            return None
    
    def get_ward_from_coordinates(self, 
                                  latitude: float, 
                                  longitude: float) -> Optional[Dict]:
        """
        Find ward/assembly constituency for given coordinates
        
        Args:
            latitude: Latitude
            longitude: Longitude
            
        Returns:
            Dict with ward information
        """
        if self.wards_gdf is None:
            logger.warning("Ward boundaries not loaded")
            return None
        
        point = Point(longitude, latitude)
        
        # Check which ward contains this point
        for idx, ward in self.wards_gdf.iterrows():
            if ward.geometry.contains(point):
                return {
                    'ward_name': ward.get('name', 'Unknown'),
                    'ward_id': idx,
                    'latitude': latitude,
                    'longitude': longitude
                }
        
        logger.warning(f"No ward found for coordinates: {latitude}, {longitude}")
        return None
    
    def get_district_from_coordinates(self,
                                     latitude: float,
                                     longitude: float) -> Optional[Dict]:
        """
        Find district for given coordinates
        
        Args:
            latitude: Latitude
            longitude: Longitude
            
        Returns:
            Dict with district information
        """
        if self.districts_gdf is None:
            logger.warning("District boundaries not loaded")
            return None
        
        point = Point(longitude, latitude)
        
        for idx, district in self.districts_gdf.iterrows():
            if district.geometry.contains(point):
                return {
                    'district_name': district.get('name', 'Unknown'),
                    'district_id': idx,
                    'latitude': latitude,
                    'longitude': longitude
                }
        
        logger.warning(f"No district found for coordinates: {latitude}, {longitude}")
        return None
    
    def get_location_details(self, 
                            latitude: float,
                            longitude: float) -> Dict:
        """
        Get comprehensive location details
        
        Args:
            latitude: Latitude
            longitude: Longitude
            
        Returns:
            Dict with all available location information
        """
        details = {
            'latitude': latitude,
            'longitude': longitude
        }
        
        # Get address
        address_info = self.reverse_geocode(latitude, longitude)
        if address_info:
            details.update(address_info)
        
        # Get ward
        ward_info = self.get_ward_from_coordinates(latitude, longitude)
        if ward_info:
            details['ward_name'] = ward_info['ward_name']
            details['ward_id'] = ward_info['ward_id']
        
        # Get district
        district_info = self.get_district_from_coordinates(latitude, longitude)
        if district_info:
            details['district_name'] = district_info['district_name']
            details['district_id'] = district_info['district_id']
        
        return details
    
    def search_location(self, query: str) -> Optional[Dict]:
        """
        Universal location search
        Handles addresses, pincodes, landmarks
        
        Args:
            query: Search query
            
        Returns:
            Dict with location details
        """
        query = query.strip()
        
        # Check if query is a pincode
        if re.match(r'^\d{6}$', query):
            geocode_result = self.geocode_pincode(query)
        else:
            geocode_result = self.geocode_address(query)
        
        if geocode_result:
            # Get full details
            details = self.get_location_details(
                geocode_result['latitude'],
                geocode_result['longitude']
            )
            
            # Add original geocode info
            details['formatted_address'] = geocode_result.get('formatted_address', '')
            
            return details
        
        return None
    
    def is_in_delhi(self, latitude: float, longitude: float) -> bool:
        """
        Check if coordinates are within Delhi boundaries
        
        Args:
            latitude: Latitude
            longitude: Longitude
            
        Returns:
            True if in Delhi
        """
        bbox = config.DELHI_BBOX
        
        return (bbox['min_lat'] <= latitude <= bbox['max_lat'] and
                bbox['min_lon'] <= longitude <= bbox['max_lon'])
    
    def is_in_gurgaon(self, latitude: float, longitude: float) -> bool:
        """
        Check if coordinates are within Gurgaon boundaries
        
        Args:
            latitude: Latitude
            longitude: Longitude
            
        Returns:
            True if in Gurgaon
        """
        bbox = config.GURGAON_BBOX
        
        return (bbox['min_lat'] <= latitude <= bbox['max_lat'] and
                bbox['min_lon'] <= longitude <= bbox['max_lon'])
    
    def find_nearest_monitoring_station(self,
                                       latitude: float,
                                       longitude: float,
                                       stations_df: pd.DataFrame,
                                       max_distance_km: float = 10) -> Optional[Dict]:
        """
        Find nearest air quality monitoring station
        
        Args:
            latitude: User latitude
            longitude: User longitude
            stations_df: DataFrame with station locations
            max_distance_km: Maximum search radius
            
        Returns:
            Dict with nearest station info and distance
        """
        from math import radians, cos, sin, asin, sqrt
        
        def haversine(lon1, lat1, lon2, lat2):
            """Calculate distance between two points on Earth"""
            lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * asin(sqrt(a))
            km = 6371 * c
            return km
        
        if stations_df is None or len(stations_df) == 0:
            return None
        
        # Calculate distances
        stations_df = stations_df.copy()
        stations_df['distance_km'] = stations_df.apply(
            lambda row: haversine(longitude, latitude, 
                                row['longitude'], row['latitude']),
            axis=1
        )
        
        # Find nearest
        nearest = stations_df.loc[stations_df['distance_km'].idxmin()]
        
        if nearest['distance_km'] <= max_distance_km:
            return {
                'station_name': nearest['location_name'],
                'latitude': nearest['latitude'],
                'longitude': nearest['longitude'],
                'distance_km': round(nearest['distance_km'], 2)
            }
        
        return None
    
    def get_ward_polygon(self, ward_name: str) -> Optional[gpd.GeoDataFrame]:
        """
        Get polygon geometry for a ward
        
        Args:
            ward_name: Name of ward
            
        Returns:
            GeoDataFrame with ward geometry
        """
        if self.wards_gdf is None:
            return None
        
        ward_gdf = self.wards_gdf[self.wards_gdf['name'] == ward_name]
        
        if len(ward_gdf) > 0:
            return ward_gdf
        
        return None


if __name__ == "__main__":
    # Example usage
    service = GeocodingService()
    
    # Test address geocoding
    print("\n1. Geocoding address:")
    result = service.search_location("Connaught Place, Delhi")
    if result:
        print(f"   Latitude: {result['latitude']}")
        print(f"   Longitude: {result['longitude']}")
        print(f"   Ward: {result.get('ward_name', 'N/A')}")
        print(f"   District: {result.get('district_name', 'N/A')}")
    
    # Test pincode
    print("\n2. Geocoding pincode:")
    result = service.search_location("110001")
    if result:
        print(f"   Location: {result.get('formatted_address', 'N/A')}")
    
    # Test reverse geocoding
    print("\n3. Reverse geocoding:")
    details = service.get_location_details(28.6139, 77.2090)
    print(f"   Address: {details.get('formatted_address', 'N/A')}")
    print(f"   Ward: {details.get('ward_name', 'N/A')}")
