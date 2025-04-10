'''import os
import ee
from datetime import datetime, timedelta
from src.data.gee_utils import initialize_gee, get_climate_data
import pandas as pd
import time

def download_climate_data(years=10, region='Global', output_dir='data/raw'):
    """Download climate data with improved error handling and yearly processing."""
    try:
        initialize_gee()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * years)
        
        # Break into yearly chunks
        yearly_dfs = []
        current_start = start_date
        while current_start < end_date:
            current_end = min(current_start + timedelta(days=365), end_date)
            print(f"Downloading data for {region} from {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}...")
            df = get_climate_data(
                current_start.strftime('%Y-%m-%d'),
                current_end.strftime('%Y-%m-%d'),
                region,
                points=100  # Reduced points for efficiency
            )
            if not df.empty:
                yearly_dfs.append(df)
            else:
                print(f"No data obtained for {region} in {current_start.year}")
            current_start = current_end
        
        if not yearly_dfs:
            print(f"No data obtained for {region} across all years")
            return None
        
        # Combine yearly data
        combined_df = pd.concat(yearly_dfs, ignore_index=True)
        
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'climate_data_{region.lower().replace(" ", "_")}.csv')
        combined_df.to_csv(output_path, index=False)
        print(f"Successfully saved data for {region} to {output_path}")
        return combined_df
        
    except Exception as e:
        print(f"Failed to download data for {region}: {str(e)}")
        return None

def batch_download_all_regions(years=10):
    """Download data for all regions."""
    regions = ['Global', 'North America', 'Europe', 'Asia', 'Africa', 'South America', 'Oceania', 'Australia', 'Antarctica']
    
    for region in regions:
        try:
            print(f"\nProcessing region: {region}...")
            download_climate_data(years=years, region=region)
            time.sleep(10)  # Avoid rate limiting
        except Exception as e:
            print(f"Failed to process {region}: {e}")
            continue

if __name__ == '__main__':
    batch_download_all_regions(years=5)'''
import os
from datetime import datetime, timedelta
import pandas as pd
import time
import logging
from typing import Optional
from src.data.gee_utils import gee_downloader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_region_data(region: str, years: int, output_dir: str) -> Optional[pd.DataFrame]:
    """Download and save climate data for a single region."""
    output_path = os.path.join(output_dir, f'climate_data_{region.lower().replace(" ", "_")}.csv')
    
    if os.path.exists(output_path):
        logger.info(f"Data exists for {region}, skipping download")
        return pd.read_csv(output_path)
    
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * years)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"DOWNLOADING {region.upper()} DATA")
        logger.info(f"Time Period: {start_date.date()} to {end_date.date()}")
        logger.info(f"{'='*60}")
        
        df = gee_downloader.get_climate_data(
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d'),
            region
        )
        
        if df.empty:
            logger.warning(f"No data obtained for {region}")
            return None
            
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df)} records to {output_path}")
        return df
        
    except Exception as e:
        logger.error(f"Failed to download {region}: {str(e)}")
        return None

def batch_download_all_regions(years: int = 5, output_dir: str = 'data/raw') -> None:
    """Download data for all regions with optimized scheduling."""
    regions = [
        'Australia', 'Oceania', 'Africa', 'South America',
        'Europe', 'Asia', 'North America', 'Antarctica', 'Global'
    ]
    
    for region in regions:
        start_time = time.time()
        try:
            df = download_region_data(region, years, output_dir)
            if df is not None:
                logger.info(f"{region} data shape: {df.shape}")
            
            # Dynamic sleep based on region size
            sleep_time = 90 if region == 'Global' else (60 if region == 'Antarctica' else 45)
            logger.info(f"Waiting {sleep_time} seconds before next region...")
            time.sleep(sleep_time)
            
            logger.info(f"{region} completed in {time.time() - start_time:.1f} seconds")
            
        except Exception as e:
            logger.error(f"Critical error processing {region}: {str(e)}")
            continue

if __name__ == '__main__':
    batch_download_all_regions(years=5)
'''import os
from datetime import datetime, timedelta
from src.data.gee_utils import get_climate_data
import pandas as pd
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_climate_data(years=5, region='Global', output_dir='data/raw'):
    """Download climate data with file checking."""
    output_path = os.path.join(output_dir, f'climate_data_{region.lower().replace(" ", "_")}.csv')
    
    if os.path.exists(output_path):
        logger.info(f"Data for {region} already exists at {output_path}. Skipping...")
        return pd.read_csv(output_path)
    
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * years)
        
        logger.info(f"============================================================\nSTARTING DOWNLOAD FOR {region.upper()}\n============================================================")
        df = get_climate_data(
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d'),
            region
        )
        
        if df.empty:
            logger.warning(f"No data obtained for {region}")
            return None
        
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Successfully saved data for {region} to {output_path}")
        return df
        
    except Exception as e:
        logger.error(f"Failed to download data for {region}: {str(e)}")
        return None

def batch_download_all_regions(years=5):
    """Download data for all regions, skipping existing files."""
    regions = ['Global', 'North America', 'Europe', 'Asia', 'Africa', 'South America', 'Oceania', 'Australia', 'Antarctica']
    
    for region in regions:
        try:
            download_climate_data(years=years, region=region)
            time.sleep(10)  # Avoid rate limiting
        except Exception as e:
            logger.error(f"Failed to process {region}: {e}")
            continue

if __name__ == '__main__':
    batch_download_all_regions(years=5)'''