'''import ee
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta

load_dotenv()

def initialize_gee():
    """Initialize Google Earth Engine with service account credentials."""
    try:
        service_account = os.getenv('GEE_SERVICE_ACCOUNT')
        private_key_path = os.getenv('GEE_SERVICE_KEY_PATH')
        credentials = ee.ServiceAccountCredentials(service_account, private_key_path)
        ee.Initialize(credentials)
    except Exception as e:
        raise Exception(f"Failed to initialize GEE: {str(e)}")

def get_climate_data(start_date, end_date, region_name, points=100):
    """Retrieve climate data from Google Earth Engine with optimized processing."""
    regions = {
        'Global': ee.Geometry.Rectangle([-180, -90, 180, 90]),
        'North America': ee.Geometry.Rectangle([-168, 12, -52, 84]),
        'Europe': ee.Geometry.Rectangle([-25, 35, 40, 72]),
        'Asia': ee.Geometry.Rectangle([60, -10, 150, 55]),
        'Africa': ee.Geometry.Rectangle([-25, -35, 55, 38]),
        'South America': ee.Geometry.Rectangle([-93, -56, -32, 15]),
        'Oceania': ee.Geometry.Rectangle([110, -50, 180, -10]),
        'Australia': ee.Geometry.Rectangle([113, -44, 154, -10]),
        'Antarctica': ee.Geometry.Rectangle([-180, -90, 180, -60])
    }
    roi = regions.get(region_name, regions['Global'])
    
    # Adjust number of points based on region (more for Global, fewer for smaller regions)
    num_points = 1000 if region_name == 'Global' else points
    
    # Convert dates to EE format
    start = ee.Date(start_date)
    end = ee.Date(end_date)
    
    # Load datasets
    modis = ee.ImageCollection('MODIS/061/MOD11A2').filterDate(start, end).filterBounds(roi)
    chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY').filterDate(start, end).filterBounds(roi)
    try:
        co = ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_CO').filterDate(start, end).select('CO_column_number_density')
    except Exception as e:
        print(f"CO data access failed: {e}")
        co = None
    
    # Generate random points
    points = ee.FeatureCollection.randomPoints(roi, num_points)
    
    def extract_values(point):
        # Use the start date as a base; we'll aggregate over time later if needed
        date = start.format('YYYY-MM-dd')
        properties = {
            'date': date,
            'temperature': modis.mean().reduceRegion(ee.Reducer.mean(), point.geometry(), 1000).get('LST_Day_1km'),
            'precipitation': chirps.mean().reduceRegion(ee.Reducer.mean(), point.geometry(), 1000).get('precipitation'),
            'latitude': point.geometry().coordinates().get(1),
            'longitude': point.geometry().coordinates().get(0)
        }
        if co:
            properties['co'] = co.mean().reduceRegion(ee.Reducer.mean(), point.geometry(), 1000).get('CO_column_number_density')
        return ee.Feature(None, properties)
    
    extracted_data = points.map(extract_values)
    data_list = []
    
    try:
        features = extracted_data.getInfo()['features']
        for feature in features:
            props = feature['properties']
            row = {
                'date': props['date'],
                'temperature': props.get('temperature'),
                'precipitation': props.get('precipitation'),
                'latitude': props['latitude'],
                'longitude': props['longitude']
            }
            if 'co' in props:
                row['co'] = props['co']
            data_list.append(row)
    except ee.EEException as e:
        raise Exception(f"GEE computation failed: {str(e)}")
    
    df = pd.DataFrame(data_list)
    if 'temperature' in df.columns:
        df['temperature'] = df['temperature'] * 0.02 - 273.15
    return df.dropna()'''
#For downloading global data
import ee
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, List, Optional, Tuple

# Configure logging
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GEEDataDownloader:
    """Robust climate data downloader with proper date handling and error recovery."""
    
    def __init__(self):
        self.initialize_gee()
        self.region_config = {
            'Global': {'points': 300, 'time_chunks': 15, 'sleep': 60},
            'Antarctica': {'points': 100, 'time_chunks': 30, 'sleep': 45},
            'default': {'points': 50, 'time_chunks': 60, 'sleep': 30}
        }
        self.datasets = {
            'temperature': ('MODIS/061/MOD11A2', 'LST_Day_1km'),
            'precipitation': ('UCSB-CHG/CHIRPS/DAILY', 'precipitation'),
            'co': ('COPERNICUS/S5P/OFFL/L3_CO', 'CO_column_number_density')
        }

    def initialize_gee(self, max_retries: int = 3) -> None:
        """Initialize GEE with exponential backoff."""
        for attempt in range(max_retries):
            try:
                credentials = ee.ServiceAccountCredentials(
                    os.getenv('GEE_SERVICE_ACCOUNT'),
                    os.getenv('GEE_SERVICE_KEY_PATH')
                )
                ee.Initialize(credentials)
                logger.info("GEE initialized successfully")
                return
            except Exception as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(f"GEE initialization failed after {max_retries} attempts")
                wait = (attempt + 1) * 15
                logger.warning(f"Attempt {attempt + 1} failed. Retrying in {wait} seconds...")
                time.sleep(wait)

    def get_region_geometry(self, region_name: str) -> ee.Geometry:
        """Get optimized geometry for each region."""
        regions = {
            'Global': ee.Geometry.Rectangle([-180, -90, 180, 90]),
            'North America': ee.Geometry.Rectangle([-168, 12, -52, 84]),
            'Europe': ee.Geometry.Rectangle([-25, 35, 40, 72]),
            'Asia': ee.Geometry.Rectangle([60, -10, 150, 55]),
            'Africa': ee.Geometry.Rectangle([-25, -35, 55, 38]),
            'South America': ee.Geometry.Rectangle([-93, -56, -32, 15]),
            'Oceania': ee.Geometry.Rectangle([110, -50, 180, -10]),
            'Australia': ee.Geometry.Rectangle([113, -44, 154, -10]),
            'Antarctica': ee.Geometry.Rectangle([-180, -90, 180, -60])
        }
        return regions.get(region_name, regions['Global'])

    def create_date_chunks(self, start_date: str, end_date: str, chunk_days: int) -> List[Tuple[str, str]]:
        """Generate date ranges with validation, capping at current date."""
        try:
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = min(datetime.strptime(end_date, '%Y-%m-%d'), datetime.now())  # Cap at today
            if start >= end:
                raise ValueError("Start date must be before end date")
            
            chunks = []
            current = start
            while current < end:
                chunk_end = min(current + timedelta(days=chunk_days), end)
                chunks.append((
                    current.strftime('%Y-%m-%d'),
                    chunk_end.strftime('%Y-%m-%d')
                ))
                current = chunk_end
            return chunks
        except Exception as e:
            logger.error(f"Date processing error: {str(e)}")
            return []

    def get_image_collection(self, dataset_name: str, date_range: Tuple[str, str], geometry: ee.Geometry) -> Optional[ee.ImageCollection]:
        """Safely load and validate image collections."""
        try:
            collection_id, band = self.datasets[dataset_name]
            collection = ee.ImageCollection(collection_id) \
                .filterDate(*date_range) \
                .filterBounds(geometry) \
                .select(band)
            
            size = collection.size().getInfo()
            if size == 0:
                logger.warning(f"No {dataset_name} data available for {date_range}")
                return None
            return collection
        except Exception as e:
            logger.error(f"Failed to load {dataset_name} data: {str(e)}")
            return None

    def process_date_chunk(self, region_name: str, date_range: Tuple[str, str]) -> List[Dict]:
        """Process a single date chunk with proper error handling."""
        config = self.region_config.get(region_name, self.region_config['default'])
        roi = self.get_region_geometry(region_name)
        
        # Load datasets
        modis = self.get_image_collection('temperature', date_range, roi)
        chirps = self.get_image_collection('precipitation', date_range, roi)
        co = self.get_image_collection('co', date_range, roi)
        
        if not any([modis, chirps, co]):  # Require at least one dataset
            logger.warning(f"No datasets available for {region_name} in {date_range}")
            return []

        # Generate points
        points = ee.FeatureCollection.randomPoints(roi, config['points'], seed=int(time.time()))
        date_str = ee.Date(date_range[0]).format('YYYY-MM-dd')
        
        def extract_values(point):
            values = {
                'date': date_str,
                'latitude': point.geometry().coordinates().get(1),
                'longitude': point.geometry().coordinates().get(0)
            }
            if modis:
                values['temperature'] = modis.mean().reduceRegion(
                    ee.Reducer.mean(), point.geometry(), 1000
                ).get(self.datasets['temperature'][1])
            if chirps:
                values['precipitation'] = chirps.mean().reduceRegion(
                    ee.Reducer.mean(), point.geometry(), 1000
                ).get(self.datasets['precipitation'][1])
            if co:
                values['co'] = co.mean().reduceRegion(
                    ee.Reducer.mean(), point.geometry(), 1000
                ).get(self.datasets['co'][1])
            return ee.Feature(None, values)
        
        try:
            extracted = points.map(extract_values)
            features = extracted.getInfo()['features']
            
            data = []
            for f in features:
                props = f['properties']
                entry = {
                    'date': props['date'],
                    'latitude': float(props['latitude']),
                    'longitude': float(props['longitude']),
                    'temperature': float(props['temperature']) * 0.02 - 273.15 if props.get('temperature') is not None else None,
                    'precipitation': float(props['precipitation']) if props.get('precipitation') is not None else None,
                    'co': float(props['co']) if props.get('co') is not None else None
                }
                data.append(entry)  # Include all points, even with partial data
            
            return data
            
        except Exception as e:
            logger.error(f"Data processing failed for {date_range}: {str(e)}")
            return []

    def get_climate_data(self, start_date: str, end_date: str, region_name: str) -> pd.DataFrame:
        """Main method to download climate data."""
        config = self.region_config.get(region_name, self.region_config['default'])
        date_chunks = self.create_date_chunks(start_date, end_date, config['time_chunks'])
        
        if not date_chunks:
            return pd.DataFrame()

        all_data = []
        logger.info(f"Processing {region_name} ({len(date_chunks)} chunks)")
        
        for i, (chunk_start, chunk_end) in enumerate(date_chunks, 1):
            for attempt in range(3):
                try:
                    logger.info(f"Chunk {i}/{len(date_chunks)}: {chunk_start} to {chunk_end}")
                    chunk_data = self.process_date_chunk(region_name, (chunk_start, chunk_end))
                    
                    if chunk_data:
                        all_data.extend(chunk_data)
                        logger.info(f"Added {len(chunk_data)} records")
                        time.sleep(config['sleep'])
                        break
                except Exception as e:
                    logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
                    if attempt == 2:
                        logger.error(f"Skipping chunk {chunk_start} to {chunk_end}")
                    time.sleep((attempt + 1) * 15)

        df = pd.DataFrame(all_data)
        if not df.empty:
            logger.info(f"Completed {region_name} with {len(df)} records")
            return df  # Don't drop NaN here to keep partial data
        logger.warning(f"No valid data obtained for {region_name}")
        return pd.DataFrame()

# Global instance for convenience
gee_downloader = GEEDataDownloader()

# Backward compatibility functions
def initialize_gee():
    gee_downloader.initialize_gee()

def get_climate_data(start_date: str, end_date: str, region_name: str) -> pd.DataFrame:
    return gee_downloader.get_climate_data(start_date, end_date, region_name)
#for downloading antarctica data and handling date properly
'''import ee
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import time
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_gee():
    """Initialize GEE with retry logic."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            service_account = os.getenv('GEE_SERVICE_ACCOUNT')
            private_key_path = os.getenv('GEE_SERVICE_KEY_PATH')
            credentials = ee.ServiceAccountCredentials(service_account, private_key_path)
            ee.Initialize(credentials)
            logger.info("GEE initialized successfully")
            return
        except Exception as e:
            if attempt == max_retries - 1:
                raise Exception(f"Failed to initialize GEE after {max_retries} attempts: {str(e)}")
            wait_time = (attempt + 1) * 10
            logger.warning(f"GEE init failed (attempt {attempt + 1}), retrying in {wait_time}s...")
            time.sleep(wait_time)

def get_region_geometry(region_name):
    """Get optimized geometry for each region."""
    regions = {
        'Global': ee.Geometry.Rectangle([-180, -90, 180, 90]),
        'North America': ee.Geometry.Rectangle([-168, 12, -52, 84]),
        'Europe': ee.Geometry.Rectangle([-25, 35, 40, 72]),
        'Asia': ee.Geometry.Rectangle([60, -10, 150, 55]),
        'Africa': ee.Geometry.Rectangle([-25, -35, 55, 38]),
        'South America': ee.Geometry.Rectangle([-93, -56, -32, 15]),
        'Oceania': ee.Geometry.Rectangle([110, -50, 180, -10]),
        'Australia': ee.Geometry.Rectangle([113, -44, 154, -10]),
        'Antarctica': ee.Geometry.Rectangle([-180, -90, 180, -60])
    }
    geometry = regions.get(region_name, regions['Global'])
    logger.debug(f"Geometry for {region_name}: {geometry.getInfo()}")
    return geometry

def get_climate_data(start_date, end_date, region_name, max_retries=3):
    """Robust climate data download with proper date handling."""
    initialize_gee()
    roi = get_region_geometry(region_name)
    
    # Configuration by region type
    config = {
        'Global': {'points': 50, 'time_chunks': 30, 'sleep_time': 30},  # Reduced points from 100 to 50
        'Antarctica': {'points': 50, 'time_chunks': 60, 'sleep_time': 20},
        'default': {'points': 30, 'time_chunks': 90, 'sleep_time': 10}
    }
    conf = config.get(region_name, config['default'])
    
    # Validate dates
    try:
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        if start_dt >= end_dt:
            raise ValueError("Start date must be before end date")
    except ValueError as e:
        logger.error(f"Invalid date format: {str(e)}")
        return pd.DataFrame()

    all_data = []
    current = start_dt
    
    while current < end_dt:
        chunk_end = min(current + timedelta(days=conf['time_chunks']), end_dt)
        range_start = current.strftime('%Y-%m-%d')
        range_end = chunk_end.strftime('%Y-%m-%d')
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Processing {region_name} from {range_start} to {range_end} (attempt {attempt + 1})")
                
                ee_start = ee.Date(range_start)
                ee_end = ee.Date(range_end)
                
                # Load datasets
                logger.info(f"Loading MODIS for {region_name}")
                modis = ee.ImageCollection('MODIS/061/MOD11A2') \
                    .filterDate(ee_start, ee_end) \
                    .filterBounds(roi)
                
                logger.info(f"Loading CHIRPS for {region_name}")
                chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY') \
                    .filterDate(ee_start, ee_end) \
                    .filterBounds(roi)
                
                logger.info(f"Generating {conf['points']} points for {region_name}")
                points = ee.FeatureCollection.randomPoints(
                    region=roi,
                    points=conf['points'],
                    seed=int(time.time())
                )
                
                # Add date to points
                def add_date(feature):
                    return feature.set('system:time_start', ee_start.millis())
                
                dated_points = points.map(add_date)
                
                # Extract values with detailed error handling
                def extract_values(feature):
                    date = ee.Date(feature.get('system:time_start')).format('YYYY-MM-dd')
                    properties = {
                        'date': date,
                        'latitude': feature.geometry().coordinates().get(1),
                        'longitude': feature.geometry().coordinates().get(0)
                    }
                    
                    try:
                        temp = modis.mean().reduceRegion(
                            ee.Reducer.mean(),
                            feature.geometry(),
                            1000
                        ).get('LST_Day_1km')
                        properties['temperature'] = ee.Number(temp) if temp is not None else None
                    except Exception as e:
                        logger.warning(f"Temperature extraction failed for point: {str(e)}")
                        properties['temperature'] = None
                    
                    try:
                        precip = chirps.mean().reduceRegion(
                            ee.Reducer.mean(),
                            feature.geometry(),
                            1000
                        ).get('precipitation')
                        properties['precipitation'] = ee.Number(precip) if precip is not None else None
                    except Exception as e:
                        logger.warning(f"Precipitation extraction failed for point: {str(e)}")
                        properties['precipitation'] = None
                    
                    return feature.set(properties)
                
                extracted = dated_points.map(extract_values)
                features = extracted.getInfo()['features']
                
                for f in features:
                    props = f['properties']
                    if props['temperature'] is not None and props['precipitation'] is not None:
                        all_data.append({
                            'date': props['date'],
                            'temperature': float(props['temperature']) * 0.02 - 273.15,
                            'precipitation': float(props['precipitation']),
                            'latitude': float(props['latitude']),
                            'longitude': float(props['longitude'])
                        })
                
                time.sleep(conf['sleep_time'])
                break
                
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    logger.error(f"Giving up on {range_start} to {range_end} after {max_retries} attempts")
                    break
                wait_time = (attempt + 1) * 15
                logger.warning(f"Retrying in {wait_time}s...")
                time.sleep(wait_time)
        
        current = chunk_end
    
    df = pd.DataFrame(all_data)
    if not df.empty:
        df = df.dropna()
        logger.info(f"Obtained {len(df)} records for {region_name}")
    else:
        logger.warning(f"No valid data obtained for {region_name}")
    return df'''