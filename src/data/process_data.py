import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import logging
from pathlib import Path

# First ensure the logs directory exists
Path('logs').mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/data_processing.log')
    ]
)
logger = logging.getLogger(__name__)

def validate_input_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean the input DataFrame."""
    required_columns = {'date', 'temperature', 'precipitation', 'latitude', 'longitude'}
    
    # Check for required columns
    missing_cols = required_columns - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Initial cleaning
    initial_count = len(df)
    df = df.dropna()
    df = df[(df['temperature'] > -50) & (df['temperature'] < 60)]
    df = df[(df['precipitation'] >= 0) & (df['precipitation'] < 1000)]
    
    logger.info(f"Data cleaning removed {initial_count - len(df)} rows ({100*(initial_count - len(df))/initial_count:.2f}%)")
    return df

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features from raw data."""
    try:
        # Date features
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day_of_year'] = df['date'].dt.dayofyear
        df['season'] = (df['month'] % 12 // 3).map({0: 'Winter', 1: 'Spring', 2: 'Summer', 3: 'Fall'})
        
        # Climate features
        if 'co2' in df.columns:
            df['co2_anomaly'] = df['co2'] - df.groupby(['latitude', 'longitude', 'month'])['co2'].transform('mean')
        
        # Temperature anomaly
        df['temp_anomaly'] = df['temperature'] - df.groupby(
            ['latitude', 'longitude', 'month']
        )['temperature'].transform('mean')
        
        return df
    except Exception as e:
        logger.error(f"Feature engineering failed: {str(e)}")
        raise

def clean_and_process_data(
    raw_data_path: str = 'data/raw/climate_data_global.csv',
    output_path: str = 'data/processed/climate_data.csv'
) -> pd.DataFrame:
    """
    Full data processing pipeline.
    
    Args:
        raw_data_path: Path to raw data CSV
        output_path: Path to save processed data
        
    Returns:
        Processed DataFrame
    """
    try:
        logger.info(f"Starting data processing from {raw_data_path}")
        
        # Ensure input exists
        if not Path(raw_data_path).exists():
            raise FileNotFoundError(f"Input file not found: {raw_data_path}")
        
        # Load and validate
        df = pd.read_csv(raw_data_path)
        logger.info(f"Initial data: {len(df)} rows, {len(df.columns)} columns")
        df = validate_input_data(df)
        
        # Feature engineering
        df = create_features(df)
        
        # Save processed data
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}")
        logger.info(f"Final data shape: {df.shape}")
        
        return df
        
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise

def prepare_training_data(
    processed_data_path: str = 'data/processed/climate_data.csv',
    test_size: float = 0.2,
    random_state: int = 42
) -> tuple:
    """
    Prepare data for model training.
    
    Returns:
        (X_train, X_test, y_train, y_test, scaler)
    """
    try:
        logger.info("Preparing training data")
        
        df = pd.read_csv(processed_data_path)
        
        # Features and target
        feature_cols = ['temperature', 'precipitation', 'latitude', 'longitude', 'year', 'month']
        if 'co2' in df.columns:
            feature_cols.append('co2')
            
        X = df[feature_cols]
        y = df['temp_anomaly']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Save scaler
        Path('models/scalers').mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, 'models/scalers/standard_scaler.joblib')
        
        logger.info("Training data prepared successfully")
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler
        
    except Exception as e:
        logger.error(f"Training data preparation failed: {str(e)}")
        raise

if __name__ == '__main__':
    try:
        # Process data
        processed_df = clean_and_process_data()
        
        # Prepare training data
        X_train, X_test, y_train, y_test, scaler = prepare_training_data()
        
        # Print summary
        print("\nData Processing Summary:")
        print(f"- Final dataset size: {len(processed_df)} samples")
        print(f"- Training set: {len(X_train)} samples")
        print(f"- Test set: {len(X_test)} samples")
        print(f"- Features used: {scaler.n_features_in_}")
        
    except Exception as e:
        logger.critical(f"Pipeline failed: {str(e)}")
        raise